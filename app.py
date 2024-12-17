from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare the dataset
data = pd.read_csv('student_data.csv')
data['Elective_Choice'] = data['Elective_Choice'].map({'Science': 0, 'Commerce': 1, 'Arts': 2})

X = data[['Math_Marks', 'Science_Marks', 'English_Marks']]
y = data['Elective_Choice']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model with increased max_iter
model = LogisticRegression(max_iter=500)

# Train the model
model.fit(X_train, y_train)

# Function to predict elective choice
def predict_elective(math_marks, science_marks, english_marks):
    # Scale the input data for prediction
    scaled_input = scaler.transform([[math_marks, science_marks, english_marks]])
    predicted_class = model.predict(scaled_input)
    electives = {0: 'Science', 1: 'Commerce', 2: 'Arts'}
    return electives[predicted_class[0]]

# Function to fetch student by ID and predict the elective choice
def fetch_student_and_predict(student_id):
    # Check if the student_id exists in the dataset
    student = data[data['Student_ID'] == student_id]
    
    if student.empty:
        return f"Student with ID {student_id} not found!"
    
    # Extract student marks
    math_marks = student['Math_Marks'].values[0]
    science_marks = student['Science_Marks'].values[0]
    english_marks = student['English_Marks'].values[0]
    
    # Predict the elective for the student
    predicted_elective = predict_elective(math_marks, science_marks, english_marks)
    
    # Return the student's data and the predicted elective
    student_info = student[['Student_ID', 'Math_Marks', 'Science_Marks', 'English_Marks']].to_dict(orient='records')[0]
    student_info['Predicted Elective'] = predicted_elective
    return student_info

@app.route('/', methods=['GET', 'POST'])
def home():
    print("Home route accessed")  # Debugging line
    result = None
    if request.method == 'POST':
        print("POST request received")  # Debugging line
        student_id = request.form['student_id']
        try:
            student_id = int(student_id)  # Convert to int
            result = fetch_student_and_predict(student_id)
            print(f"Prediction result: {result}")  # Debugging line
        except ValueError:
            result = "Please enter a valid Student ID."
            print(result)  # Debugging line
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
