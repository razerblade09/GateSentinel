import sqlite3
from datetime import datetime, timedelta

def create_expired_guest():
    db_path = r'C:\Users\varga\OneDrive\Documents\Latest_LPR\system\admin\license_plate_system.db'
    
    guest_data = {
        'plate_number': 'NGP3944',
        'access_status': 'Expired',
        # Set first detection to 48 hours ago (definitely expired)
        'first_detection_time': (datetime.now() - timedelta(hours=48)).strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Remove existing entry if exists
        cursor.execute('DELETE FROM guests WHERE plate_number = ?', (guest_data['plate_number'],))
        
        # Insert expired guest
        cursor.execute('''
            INSERT INTO guests (
                plate_number,
                access_status,
                first_detection_time
            ) VALUES (?, ?, ?)
        ''', (
            guest_data['plate_number'],
            guest_data['access_status'],
            guest_data['first_detection_time']
        ))

        conn.commit()
        print(f"Successfully created expired guest:")
        print(f"Plate: {guest_data['plate_number']}")
        print(f"Status: {guest_data['access_status']}")
        print(f"First Detected: {guest_data['first_detection_time']}")

        # Verify entry
        cursor.execute('''
            SELECT plate_number, access_status,
                   julianday('now') - julianday(first_detection_time) as hours_since_detection
            FROM guests 
            WHERE plate_number = ?
        ''', (guest_data['plate_number'],))
        
        result = cursor.fetchone()
        if result:
            print("\nVerification:")
            print(f"Plate: {result[0]}")
            print(f"Status: {result[1]}")
            print(f"Hours Since Detection: {result[2]*24:.1f} hours")

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_expired_guest()