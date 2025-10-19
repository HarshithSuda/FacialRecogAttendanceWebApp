import os
import logging
from datetime import datetime, timedelta, date
from flask import render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from sqlalchemy import func, and_, or_

from app import app, db
from models import User, Student, ClassSession, Attendance, ProcessedImage
from facial_recognition import add_student_with_face, process_class_photo, process_multiple_photos

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Ensure upload directories exist
def ensure_dirs():
    for folder in ['uploads', 'processed', 'students']:
        directory = os.path.join(app.static_folder, folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

# Set upload folder configuration
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(app.static_folder, 'processed')
app.config['STUDENTS_FOLDER'] = os.path.join(app.static_folder, 'students')

# Create directories if they don't exist
ensure_dirs()

# Helper functions
def allowed_file(filename):
    """Check if the file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = User.query.filter(
            or_(User.username == username, User.email == email)
        ).first()
        
        if existing_user:
            flash('Username or email already exists', 'danger')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        # Make the first registered user an admin
        if User.query.count() == 0:
            user.is_admin = True
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Main application routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent class sessions
    recent_sessions = ClassSession.query.order_by(ClassSession.date.desc()).limit(5).all()
    
    # Get attendance statistics
    today = date.today()
    week_ago = today - timedelta(days=7)
    
    today_attendance = ClassSession.query.filter(ClassSession.date == today).all()
    week_attendance = ClassSession.query.filter(ClassSession.date >= week_ago).all()
    
    total_students = Student.query.count()
    total_sessions = ClassSession.query.count()
    
    # Calculate attendance percentages
    today_percentage = 0
    week_percentage = 0
    
    if today_attendance:
        present_today = sum(session.present_students for session in today_attendance)
        total_today = sum(session.total_students for session in today_attendance)
        today_percentage = (present_today / total_today * 100) if total_today > 0 else 0
    
    if week_attendance:
        present_week = sum(session.present_students for session in week_attendance)
        total_week = sum(session.total_students for session in week_attendance)
        week_percentage = (present_week / total_week * 100) if total_week > 0 else 0
    
    return render_template('dashboard.html', 
                          recent_sessions=recent_sessions,
                          total_students=total_students,
                          total_sessions=total_sessions,
                          today_percentage=today_percentage,
                          week_percentage=week_percentage)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        # Check if the form contains the class session information
        session_name = request.form.get('session_name')
        session_date = request.form.get('session_date')
        session_course = request.form.get('session_course', '')
        
        # Validate session data
        if not session_name or not session_date:
            flash('Session name and date are required', 'danger')
            return redirect(url_for('upload'))
        
        try:
            session_date = datetime.strptime(session_date, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format', 'danger')
            return redirect(url_for('upload'))
        
        # Create a new class session
        class_session = ClassSession(
            name=session_name,
            date=session_date,
            course=session_course,
            creator_id=current_user.id
        )
        db.session.add(class_session)
        db.session.flush()  # Get the session ID without committing
        
        # Check if files were uploaded
        if 'files' not in request.files:
            flash('No files selected', 'danger')
            db.session.rollback()
            return redirect(url_for('upload'))
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            flash('No files selected', 'danger')
            db.session.rollback()
            return redirect(url_for('upload'))
        
        # Filter valid image files
        valid_files = [f for f in files if f and allowed_file(f.filename)]
        if not valid_files:
            flash('No valid image files selected', 'danger')
            db.session.rollback()
            return redirect(url_for('upload'))
        
        # Process all valid files for attendance
        success, results = process_multiple_photos(valid_files, class_session.id)
        
        if success:
            db.session.commit()
            flash('Images processed successfully!', 'success')
            return redirect(url_for('attendance', session_id=class_session.id))
        else:
            db.session.rollback()
            flash('Error processing images', 'danger')
            return redirect(url_for('upload'))
    
    return render_template('upload.html')

@app.route('/students', methods=['GET', 'POST'])
@login_required
def students():
    if request.method == 'POST':
        # Check if it's an Excel upload
        excel_file = request.files.get('excel_file')
        if excel_file and excel_file.filename.endswith(('.xlsx', '.xls')):
            try:
                return import_students_from_excel(excel_file)
            except Exception as e:
                flash(f'Error importing from Excel: {str(e)}', 'danger')
                return redirect(url_for('students'))
        
        # Regular student addition
        roll_number = request.form.get('roll_number')
        name = request.form.get('name')
        email = request.form.get('email', '')
        course = request.form.get('course', '')
        student_photo = request.files.get('student_photo')
        
        # Validate student data
        if not roll_number or not name:
            flash('Roll number and name are required', 'danger')
            return redirect(url_for('students'))
        
        # Check if student already exists
        existing_student = Student.query.filter_by(roll_number=roll_number).first()
        if existing_student:
            flash('A student with this roll number already exists', 'danger')
            return redirect(url_for('students'))
        
        # Validate and process the photo
        if not student_photo or student_photo.filename == '':
            flash('Student photo is required', 'danger')
            return redirect(url_for('students'))
        
        if not allowed_file(student_photo.filename):
            flash('Invalid file format. Please upload a .jpg, .jpeg, or .png file', 'danger')
            return redirect(url_for('students'))
        
        # Create the student and add their face
        success, message = add_student_with_face(roll_number, name, student_photo)
        
        if success:
            # Update additional student information
            student = Student.query.filter_by(roll_number=roll_number).first()
            student.email = email
            student.course = course
            db.session.commit()
            
            flash(message, 'success')
        else:
            flash(message, 'danger')
        
        return redirect(url_for('students'))
    
    # Get all students for the listing
    all_students = Student.query.order_by(Student.roll_number).all()
    return render_template('students.html', students=all_students)

def import_students_from_excel(excel_file):
    """Import students from an Excel file with roll numbers, names, and photo paths"""
    import pandas as pd
    import requests
    from io import BytesIO
    from PIL import Image
    import os
    import tempfile
    
    # Create a temporary directory to store downloaded images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Check if the required columns exist
        required_columns = ['roll_number', 'name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            flash(f"Missing columns in Excel file: {', '.join(missing_columns)}", 'danger')
            return redirect(url_for('students'))
        
        # Track import stats
        total_rows = len(df)
        successful_imports = 0
        failed_imports = 0
        
        for _, row in df.iterrows():
            try:
                roll_number = str(row['roll_number'])
                name = row['name']
                
                # Check if student already exists
                existing_student = Student.query.filter_by(roll_number=roll_number).first()
                if existing_student:
                    logger.warning(f"Student with roll number {roll_number} already exists, skipping")
                    failed_imports += 1
                    continue
                
                # Create a new student without face encoding if photo_path is not provided
                if 'photo_path' not in df.columns or pd.isna(row['photo_path']):
                    # Create student without face encoding
                    student = Student(roll_number=roll_number, name=name)
                    if 'email' in df.columns and not pd.isna(row['email']):
                        student.email = row['email']
                    if 'course' in df.columns and not pd.isna(row['course']):
                        student.course = row['course']
                    
                    db.session.add(student)
                    db.session.commit()
                    successful_imports += 1
                    continue
                
                # If we reach here, photo_path is provided
                photo_path = row['photo_path']
                
                # Process the photo
                # Check if it's a URL or local path
                if photo_path.startswith(('http://', 'https://')):
                    # It's a URL, download the image
                    try:
                        response = requests.get(photo_path)
                        if response.status_code != 200:
                            logger.error(f"Failed to download image from {photo_path}")
                            # Create student without face encoding
                            student = Student(roll_number=roll_number, name=name)
                            if 'email' in df.columns and not pd.isna(row['email']):
                                student.email = row['email']
                            if 'course' in df.columns and not pd.isna(row['course']):
                                student.course = row['course']
                            
                            db.session.add(student)
                            db.session.commit()
                            successful_imports += 1
                            continue
                        
                        # Save to a temporary file
                        img_temp = os.path.join(temp_dir, f"{roll_number}_temp.jpg")
                        with open(img_temp, 'wb') as f:
                            f.write(response.content)
                        
                        photo_path = img_temp
                    except Exception as e:
                        logger.error(f"Error downloading image from {photo_path}: {str(e)}")
                        # Create student without face encoding
                        student = Student(roll_number=roll_number, name=name)
                        if 'email' in df.columns and not pd.isna(row['email']):
                            student.email = row['email']
                        if 'course' in df.columns and not pd.isna(row['course']):
                            student.course = row['course']
                        
                        db.session.add(student)
                        db.session.commit()
                        successful_imports += 1
                        continue
                
                try:
                    # Check if file exists at the specified path
                    if not os.path.exists(photo_path):
                        logger.warning(f"Image file not found at path: {photo_path}")
                        # Create student without face encoding
                        student = Student(roll_number=roll_number, name=name)
                        if 'email' in df.columns and not pd.isna(row['email']):
                            student.email = row['email']
                        if 'course' in df.columns and not pd.isna(row['course']):
                            student.course = row['course']
                        
                        db.session.add(student)
                        db.session.commit()
                        successful_imports += 1
                        continue
                    
                    # Create a custom file-like object for add_student_with_face
                    class ImageFile:
                        def __init__(self, path):
                            self.path = path
                            self.filename = os.path.basename(path)
                            
                        def read(self):
                            with open(self.path, 'rb') as f:
                                return f.read()
                        
                        def save(self, destination):
                            import shutil
                            shutil.copy2(self.path, destination)
                    
                    # Create our custom file object
                    image_file = ImageFile(photo_path)
                    
                    # Add the student with the face
                    success, message = add_student_with_face(roll_number, name, image_file)
                except Exception as e:
                    logger.error(f"Error processing image for {roll_number}: {str(e)}")
                    # Create student without face encoding
                    student = Student(roll_number=roll_number, name=name)
                    if 'email' in df.columns and not pd.isna(row['email']):
                        student.email = row['email']
                    if 'course' in df.columns and not pd.isna(row['course']):
                        student.course = row['course']
                    
                    db.session.add(student)
                    db.session.commit()
                    successful_imports += 1
                    continue
                
                if success:
                    successful_imports += 1
                    # Add additional info if available in the Excel
                    student = Student.query.filter_by(roll_number=roll_number).first()
                    if 'email' in df.columns and not pd.isna(row['email']):
                        student.email = row['email']
                    if 'course' in df.columns and not pd.isna(row['course']):
                        student.course = row['course']
                    db.session.commit()
                else:
                    logger.error(f"Failed to add student {roll_number}: {message}")
                    failed_imports += 1
            
            except Exception as e:
                logger.exception(f"Error processing row for {row.get('roll_number', 'unknown')}: {str(e)}")
                failed_imports += 1
        
        # Report the results
        if successful_imports > 0:
            flash(f"Successfully imported {successful_imports} out of {total_rows} students", 'success')
        
        if failed_imports > 0:
            flash(f"Failed to import {failed_imports} students. Check the logs for details.", 'warning')
        
        return redirect(url_for('students'))
    
    except Exception as e:
        logger.exception(f"Excel import error: {str(e)}")
        flash(f"Error importing from Excel: {str(e)}", 'danger')
        return redirect(url_for('students'))
    finally:
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/attendance')
@login_required
def attendance():
    session_id = request.args.get('session_id')
    
    # If a specific session is requested
    if session_id:
        try:
            session_id = int(session_id)
            class_session = ClassSession.query.get_or_404(session_id)
            
            # Get attendance records for this session
            attendance_records = Attendance.query.filter_by(class_session_id=session_id).all()
            
            # Get the processed images for this session
            processed_images = ProcessedImage.query.filter_by(class_session_id=session_id).all()
            
            # Get students present in this session
            present_students = Student.query.join(Attendance).filter(
                Attendance.class_session_id == session_id
            ).all()
            
            # Get all students to identify who is absent
            all_students = Student.query.all()
            absent_students = [s for s in all_students if s not in present_students]
            
            return render_template('attendance.html', 
                                  class_session=class_session,
                                  present_students=present_students,
                                  absent_students=absent_students,
                                  processed_images=processed_images,
                                  attendance_mode='session')
        except (ValueError, TypeError):
            flash('Invalid session ID', 'danger')
            # Fall through to date range view
    
    # Otherwise show all sessions for date range selection
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Default to last 7 days if no dates provided
    if not start_date or not end_date:
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
    else:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format', 'danger')
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
    
    # Get sessions in the date range
    sessions = ClassSession.query.filter(
        ClassSession.date.between(start_date, end_date)
    ).order_by(ClassSession.date.desc()).all()
    
    # Calculate overall attendance stats for the period
    total_students = Student.query.count()
    total_attendance = sum(session.present_students for session in sessions)
    total_possible = sum(session.total_students for session in sessions)
    attendance_percentage = (total_attendance / total_possible * 100) if total_possible > 0 else 0
    
    return render_template('attendance.html',
                          sessions=sessions,
                          start_date=start_date,
                          end_date=end_date,
                          total_students=total_students,
                          attendance_percentage=attendance_percentage,
                          attendance_mode='range')

@app.route('/mark_manual_attendance', methods=['POST'])
@login_required
def mark_manual_attendance():
    """Handle manual attendance entries from the form"""
    try:
        roll_number = request.form.get('roll_number')
        status = request.form.get('status', 'present')  # default to 'present'
        class_session_id = request.form.get('class_session_id')
        
        if not roll_number or not class_session_id:
            flash('Missing required fields', 'danger')
            return redirect(url_for('attendance', session_id=class_session_id))
        
        # Find the student by roll number
        student = Student.query.filter_by(roll_number=roll_number).first()
        
        if not student:
            flash(f'Student with roll number {roll_number} not found', 'danger')
            return redirect(url_for('attendance', session_id=class_session_id))
        
        # Check if this student already has an attendance record for this session
        attendance_record = Attendance.query.filter_by(
            student_id=student.id,
            class_session_id=class_session_id
        ).first()
        
        if status == 'absent' and attendance_record:
            # If marking as absent and record exists, delete it
            db.session.delete(attendance_record)
            flash(f'{student.name} has been marked as absent', 'success')
        elif status in ['present', 'late'] and not attendance_record:
            # If marking as present/late and no record exists, create one
            attendance_record = Attendance(
                student_id=student.id,
                class_session_id=class_session_id,
                status=status
            )
            db.session.add(attendance_record)
            flash(f'{student.name} has been marked as {status}', 'success')
        elif attendance_record and attendance_record.status != status:
            # If record exists but status is different, update it
            attendance_record.status = status
            flash(f'Status for {student.name} has been updated to {status}', 'success')
        else:
            # No changes needed
            flash(f'No changes made to attendance for {student.name}', 'info')
        
        # Update class session stats
        class_session = ClassSession.query.get(class_session_id)
        if class_session:
            total_students = Student.query.count()
            present_students = Attendance.query.filter_by(class_session_id=class_session_id).count()
            class_session.total_students = total_students
            class_session.present_students = present_students
        
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error marking manual attendance: {str(e)}")
        flash(f"Error: {str(e)}", 'danger')
    
    return redirect(url_for('attendance', session_id=class_session_id))

@app.route('/reports')
@login_required
def reports():
    # Get date range parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    report_type = request.args.get('report_type', 'daily')
    
    # Default to last 30 days if no dates provided
    if not start_date or not end_date:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
    else:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format', 'danger')
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
    
    # Get basic data for the period
    sessions = ClassSession.query.filter(
        ClassSession.date.between(start_date, end_date)
    ).order_by(ClassSession.date).all()
    
    # For student-specific attendance
    students = Student.query.all()
    
    # Calculate student attendance percentages
    student_attendance = {}
    for student in students:
        present_count = Attendance.query.join(ClassSession).filter(
            Attendance.student_id == student.id,
            ClassSession.date.between(start_date, end_date)
        ).count()
        
        total_sessions = len(sessions)
        attendance_percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0
        
        student_attendance[student.id] = {
            'roll_number': student.roll_number,
            'name': student.name,
            'present_count': present_count,
            'total_sessions': total_sessions,
            'percentage': attendance_percentage
        }
    
    # Daily attendance data for charts
    daily_data = {}
    for session in sessions:
        date_str = session.date.strftime('%Y-%m-%d')
        if date_str not in daily_data:
            daily_data[date_str] = {
                'present': 0, 
                'total': 0
            }
        
        daily_data[date_str]['present'] += session.present_students
        daily_data[date_str]['total'] += session.total_students
    
    # Course-wise attendance data if available
    course_data = {}
    if any(session.course for session in sessions):
        for session in sessions:
            if not session.course:
                continue
                
            if session.course not in course_data:
                course_data[session.course] = {
                    'present': 0,
                    'total': 0
                }
            
            course_data[session.course]['present'] += session.present_students
            course_data[session.course]['total'] += session.total_students
    
    # Track students with 3+ consecutive absences
    students_with_consecutive_absences = []
    
    if len(sessions) >= 3:  # Only check if we have at least 3 sessions
        # Sort sessions by date
        sorted_sessions = sorted(sessions, key=lambda s: s.date)
        
        for student in students:
            max_consecutive_absences = 0
            current_consecutive_absences = 0
            consecutive_sessions = []
            
            for session in sorted_sessions:
                # Check if student was absent in this session
                attendance = Attendance.query.filter_by(
                    student_id=student.id,
                    class_session_id=session.id
                ).first()
                
                if not attendance:  # Student was absent
                    current_consecutive_absences += 1
                    consecutive_sessions.append(session)
                    
                    if current_consecutive_absences > max_consecutive_absences:
                        max_consecutive_absences = current_consecutive_absences
                else:  # Student was present, reset counter
                    current_consecutive_absences = 0
                    consecutive_sessions = []
            
            if max_consecutive_absences >= 3:
                students_with_consecutive_absences.append({
                    'student': student,
                    'consecutive_absences': max_consecutive_absences,
                    'sessions': consecutive_sessions[:3]  # Show just the first 3 sessions
                })
    
    return render_template('reports.html',
                          start_date=start_date,
                          end_date=end_date,
                          report_type=report_type,
                          sessions=sessions,
                          students=students,
                          student_attendance=student_attendance,
                          daily_data=daily_data,
                          course_data=course_data,
                          students_with_consecutive_absences=students_with_consecutive_absences)

# API routes for AJAX calls
@app.route('/api/attendance_data')
@login_required
def api_attendance_data():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
    
    # Query attendance data for the date range
    sessions = ClassSession.query.filter(
        ClassSession.date.between(start_date, end_date)
    ).order_by(ClassSession.date).all()
    
    # Format data for chart.js
    labels = []
    present_data = []
    absent_data = []
    
    for session in sessions:
        labels.append(session.date.strftime('%Y-%m-%d'))
        present_data.append(session.present_students)
        absent_data.append(session.total_students - session.present_students)
    
    return jsonify({
        'labels': labels,
        'present': present_data,
        'absent': absent_data
    })

@app.route('/api/student_attendance/<int:student_id>')
@login_required
def api_student_attendance(student_id):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
    
    # Get the student
    student = Student.query.get_or_404(student_id)
    
    # Get all sessions in the date range
    sessions = ClassSession.query.filter(
        ClassSession.date.between(start_date, end_date)
    ).order_by(ClassSession.date).all()
    
    # Check attendance for each session
    attendance_data = []
    for session in sessions:
        attendance = Attendance.query.filter_by(
            student_id=student_id,
            class_session_id=session.id
        ).first()
        
        attendance_data.append({
            'date': session.date.strftime('%Y-%m-%d'),
            'session_name': session.name,
            'status': attendance.status if attendance else 'absent'
        })
    
    return jsonify({
        'student': {
            'id': student.id,
            'roll_number': student.roll_number,
            'name': student.name
        },
        'attendance': attendance_data
    })

# Static file serving routes
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
@login_required
def processed_file(filename):
    return send_from_directory(current_app.config['PROCESSED_FOLDER'], filename)

@app.route('/students/<filename>')
@login_required
def student_file(filename):
    return send_from_directory(current_app.config['STUDENTS_FOLDER'], filename)
