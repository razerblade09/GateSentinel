from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
from datetime import datetime, timedelta
import re
import pandas as pd
from io import BytesIO
import pytz  # For timezone handling
import os
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# Define the absolute path to the database
DB_PATH = os.path.join(os.path.dirname(__file__), 'license_plate_system.db')

# Global variable to track the total number of guests
guest_counter = 0

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Function to expire old guests
def expire_old_guests():
    with get_db() as conn:
        conn.execute('''
            UPDATE guests
            SET access_status = 'Expired'
            WHERE access_status = 'Active'
            AND datetime(first_detection_time) < datetime('now', '-24 hours')
        ''')
        conn.commit()
    print("Expired old guests.")

# Schedule the task to run every hour
scheduler = BackgroundScheduler()
scheduler.add_job(expire_old_guests, 'interval', hours=1)
scheduler.start()

@app.route('/')
def admin_panel():
    return render_template('index.html')

@app.route('/api/cars', methods=['GET'])
def get_cars():
    with get_db() as conn:
        # Update database to permanently set expired status
        conn.execute('''
            UPDATE cars
            SET status = 'Expired'
            WHERE expiration_date < CURRENT_TIMESTAMP
            AND status != 'Expired'
        ''')
        conn.commit()

        # Fetch and return updated car list
        cars = conn.execute('''
            SELECT *, 
            CASE 
                WHEN expiration_date < CURRENT_TIMESTAMP THEN 'Expired'
                ELSE status 
            END AS current_status
            FROM cars
        ''').fetchall()
        
        return jsonify([dict(row) for row in cars])

@app.route('/api/guests', methods=['POST'])
def add_guest():
    global guest_counter
    data = request.json
    with get_db() as conn:
        try:
            conn.execute('''
                INSERT INTO guests (plate_number, access_status)
                VALUES (?, ?)
            ''', (data['plate_number'], 'Active'))
            conn.commit()
            guest_counter += 1
            return jsonify({'success': True, 'guest_count': guest_counter})
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Guest already exists'}), 400

@app.route('/api/guests/<plate>', methods=['GET'])
def get_guest(plate):
    with get_db() as conn:
        guest = conn.execute('SELECT * FROM guests WHERE plate_number = ?', (plate,)).fetchone()
        if guest:
            return jsonify(dict(guest))
        return jsonify({'error': 'Guest not found'}), 404

@app.route('/api/guests/<plate>/expire', methods=['PUT'])
def expire_guest(plate):
    with get_db() as conn:
        conn.execute('''
            UPDATE guests SET
            access_status = 'Expired'
            WHERE plate_number = ?
        ''', (plate,))
        conn.commit()
    return jsonify({'success': True})

@app.route('/api/guests/<plate>/register', methods=['PUT'])
def register_guest(plate):
    data = request.json
    with get_db() as conn:
        # Insert the guest into the cars table as a registered vehicle
        conn.execute('''
            INSERT INTO cars 
            (plate_number, owner_name, car_model, car_color, status, expiration_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            plate,
            data['owner'],
            data['model'],
            data['color'],
            'Registered',
            (datetime.now() + timedelta(days=395)).strftime('%Y-%m-%d %H:%M:%S')
        ))
        # Remove the guest from the guests table
        conn.execute('DELETE FROM guests WHERE plate_number = ?', (plate,))
        conn.commit()
    return jsonify({'success': True})

@app.route('/api/guests/check-access', methods=['GET'])
def check_guest_access():
    plate = request.args.get('plate')
    with get_db() as conn:
        guest = conn.execute('''
            SELECT * FROM guests 
            WHERE plate_number = ? 
            AND access_status = 'Active'
            AND datetime(first_detection_time) >= datetime('now', '-24 hours')
        ''', (plate,)).fetchone()
        if guest:
            return jsonify({'access': True})
        else:
            return jsonify({'access': False})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Fetch logs from the database, optionally filtered by plate number."""
    plate_number = request.args.get('plate')
    with get_db() as conn:
        query = 'SELECT * FROM logs'
        if plate_number:
            query += ' WHERE plate_number = ? ORDER BY scan_time DESC'
            logs = conn.execute(query, (plate_number,)).fetchall()
        else:
            query += ' ORDER BY scan_time DESC'
            logs = conn.execute(query).fetchall()
        return jsonify([dict(row) for row in logs])

@app.route('/api/logs', methods=['POST'])
def add_log():
    global guest_counter
    data = request.json
    local_tz = pytz.timezone('Asia/Manila')  # Set your timezone
    local_time = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    with get_db() as conn:
        # Check if the plate number belongs to a registered vehicle
        car = conn.execute('''
            SELECT * FROM cars 
            WHERE plate_number = ? 
        ''', (data['plate_number'],)).fetchone()

        # Check if the plate number belongs to a guest
        guest = conn.execute('''
            SELECT * FROM guests 
            WHERE plate_number = ? 
        ''', (data['plate_number'],)).fetchone()

        # Determine the status
        if car:
            status = car['status']  # Use the status from the cars table
        elif guest:
            if guest['access_status'] == 'Expired':
                status = 'Guest (Expired)'
            else:
                status = 'Guest'
        else:
            status = 'Detected'  # Default status for unknown plates

        # Check for recent logs within 5 minutes
        recent_log = conn.execute('''
            SELECT 1 FROM logs 
            WHERE plate_number = ? 
            AND datetime(scan_time) >= datetime('now', '-5 minutes', 'localtime')
            LIMIT 1
        ''', (data['plate_number'],)).fetchone()

        if recent_log:
            return jsonify({'success': False, 'error': 'Cooldown active'}), 400

        # Get the last log type for this plate (regardless of day)
        last_log = conn.execute('''
            SELECT type FROM logs 
            WHERE plate_number = ? 
            ORDER BY scan_time DESC 
            LIMIT 1
        ''', (data['plate_number'],)).fetchone()

        # Determine the next log type
        if status in ['Expired', 'Guest (Expired)']:
            next_type = 'Access Denied'
        else:
            if last_log:
                last_type = last_log['type']
                next_type = 'Time Out' if last_type == 'Time In' else 'Time In'
            else:
                # If no previous logs, default to Time In
                next_type = 'Time In'

        # Insert new log with the determined type, updated status, and guest count
        conn.execute('''
            INSERT INTO logs (plate_number, status, scan_time, type, guest_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['plate_number'], status, local_time, next_type, guest_counter))
        conn.commit()
    return jsonify({'success': True})

@app.route('/api/logs/download', methods=['GET'])
def download_logs():
    """Download logs as an Excel file."""
    with get_db() as conn:
        logs = conn.execute('SELECT * FROM logs ORDER BY scan_time DESC').fetchall()
        logs_df = pd.DataFrame(logs, columns=["id", "plate_number", "scan_time", "status", "type", "guest_count"])
        
        # Create an in-memory Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            logs_df.to_excel(writer, index=False, sheet_name='Logs')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='logs.xlsx'
        )

@app.route('/api/cars', methods=['POST'])
def add_car():
    data = request.json
    
    # Clean and validate plate number
    clean_plate = re.sub(r'\s+', '', data['plate']).upper()
    if not clean_plate or len(clean_plate) < 3:
        return jsonify({'error': 'Invalid license plate'}), 400

    expiration_date = (datetime.now() + timedelta(days=395)).strftime('%Y-%m-%d %H:%M:%S')
    
    with get_db() as conn:
        try:
            conn.execute('''
                INSERT INTO cars 
                (plate_number, owner_name, car_model, car_color, status, expiration_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                clean_plate,
                data['owner'],
                data['model'],
                data['color'],
                'Registered',
                expiration_date
            ))
            conn.commit()
            return jsonify({'success': True})
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Plate already exists'}), 400

@app.route('/api/cars/<plate>', methods=['DELETE'])
def delete_car(plate):
    with get_db() as conn:
        conn.execute('DELETE FROM cars WHERE plate_number = ?', (plate,))
        conn.commit()
        return jsonify({'success': True})

@app.route('/api/cars/<plate>', methods=['PUT'])
def update_car(plate):
    data = request.json
    with get_db() as conn:
        conn.execute('''
            UPDATE cars SET
            owner_name = ?,
            car_model = ?,
            car_color = ?
            WHERE plate_number = ?
        ''', (data['owner'], data['model'], data['color'], plate))
        conn.commit()
    return jsonify({'success': True})

@app.route('/api/cars/<plate>/renew', methods=['PUT'])
def renew_car(plate):
    new_expiration = (datetime.now() + timedelta(days=395)).strftime('%Y-%m-%d %H:%M:%S')
    with get_db() as conn:
        conn.execute('''
            UPDATE cars SET
            expiration_date = ?,
            status = 'Registered'
            WHERE plate_number = ?
        ''', (new_expiration, plate))
        conn.commit()
    return jsonify({'success': True})

@app.route('/api/cars/<plate>', methods=['GET'])
def get_car(plate):
    with get_db() as conn:
        car = conn.execute('SELECT * FROM cars WHERE plate_number = ?', (plate,)).fetchone()
        if car:
            return jsonify(dict(car))
        return jsonify({'error': 'Vehicle not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)