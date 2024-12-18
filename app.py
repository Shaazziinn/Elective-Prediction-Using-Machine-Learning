from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Mock user database (for login functionality)
users = {
    "admin": "password123",  # Replace with hashed passwords in production
}

# Load and prepare the dataset
data = pd.read_csv('ElectivePrediction/student_data.csv')
data['Elective_Choice'] = data['Elective_Choice'].map({'Science': 0, 'Commerce': 1, 'Arts': 2})

X = data[['Math_Marks', 'Science_Marks', 'English_Marks']]
y = data['Elective_Choice']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Function to predict elective choice
def predict_elective(math_marks, science_marks, english_marks):
    scaled_input = scaler.transform([[math_marks, science_marks, english_marks]])
    predicted_class = model.predict(scaled_input)
    electives = {0: 'Science', 1: 'Commerce', 2: 'Arts'}
    return electives[predicted_class[0]]

def fetch_student_and_predict(student_id):
    student = data[data['Student_ID'] == student_id]
    if student.empty:
        return f"Student with ID {student_id} not found!"
    
    math_marks = student['Math_Marks'].values[0]
    science_marks = student['Science_Marks'].values[0]
    english_marks = student['English_Marks'].values[0]
    
    predicted_elective = predict_elective(math_marks, science_marks, english_marks)
    student_info = student[['Student_ID', 'Math_Marks', 'Science_Marks', 'English_Marks']].to_dict(orient='records')[0]
    student_info['Predicted Elective'] = predicted_elective
    return student_info

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            flash('You have successfully logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have successfully logged out!', 'info')
    return redirect(url_for('login'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    result = None
    if request.method == 'POST':
        student_id = request.form['student_id']
        try:
            student_id = int(student_id)
            result = fetch_student_and_predict(student_id)
        except ValueError:
            result = "Please enter a valid Student ID."
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
