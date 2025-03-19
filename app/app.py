import uvicorn
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import joblib
from pydantic import BaseModel, Field


app = FastAPI()
default_classifier = joblib.load("models/model.sav")


# Define the homepage with a form
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Loan Prediction Service</title>
        </head>
        <body>
            <h1>Welcome to the Loan Prediction Service</h1>
            <form action="/evaluate" method="post">
                <label>DAYS_LAST_PHONE_CHANGE:</label>
                <input type="number" name="DAYS_LAST_PHONE_CHANGE" placeholder="0 - 10,000" min="0" max="10000"><br>

                <label>TOTALAREA_MODE:</label>
                <input type="number" name="TOTALAREA_MODE" step="0.01" min="0" max="1" placeholder="0 - 1"><br>

                <label>WEEKDAY_APPR_PROCESS_START:</label>
                <select name="WEEKDAY_APPR_PROCESS_START">
                    <option value="">None</option>
                    <option value="MONDAY">MONDAY</option>
                    <option value="TUESDAY">TUESDAY</option>
                    <option value="WEDNESDAY">WEDNESDAY</option>
                    <option value="THURSDAY">THURSDAY</option>
                    <option value="FRIDAY">FRIDAY</option>
                    <option value="SATURDAY">SATURDAY</option>
                    <option value="SUNDAY">SUNDAY</option>
                </select><br>

                <label>AMT_ANNUITY:</label>
                <input type="number" name="AMT_ANNUITY" placeholder="Enter amount" min="0"><br>

                <label>AMT_CREDIT:</label>
                <input type="number" name="AMT_CREDIT" placeholder="Enter credit amount" min="0"><br>

                <label>EXT_SOURCE_2:</label>
                <input type="number" name="EXT_SOURCE_2" step="0.01" min="0" max="1" placeholder="0 - 1"><br>

                <label>EXT_SOURCE_3:</label>
                <input type="number" name="EXT_SOURCE_3" step="0.01" min="0" max="1" placeholder="0 - 1"><br>

                <label>OCCUPATION_TYPE:</label>
                <select name="OCCUPATION_TYPE">
                    <option value="">None</option>
                    <option value="Low-skill Laborers">Low-skill Laborers</option>
                    <option value="Drivers">Drivers</option>
                    <option value="Sales staff">Sales staff</option>
                    <option value="High skill tech staff">High skill tech staff</option>
                    <option value="Core staff">Core staff</option>
                    <option value="Laborers">Laborers</option>
                    <option value="Managers">Managers</option>
                    <option value="Accountants">Accountants</option>
                    <option value="Medicine staff">Medicine staff</option>
                    <option value="Security staff">Security staff</option>
                    <option value="Private service staff">Private service staff</option>
                    <option value="Secretaries">Secretaries</option>
                    <option value="Cleaning staff">Cleaning staff</option>
                    <option value="Cooking staff">Cooking staff</option>
                    <option value="HR staff">HR staff</option>
                    <option value="Waiters/barmen staff">Waiters/barmen staff</option>
                    <option value="Realty agents">Realty agents</option>
                    <option value="IT staff">IT staff</option>
                </select><br>

                <label>ORGANIZATION_TYPE:</label>
                <select name="ORGANIZATION_TYPE">
                    <option value="">None</option>
                    <option value="Kindergarten">Kindergarten</option>
                    <option value="Self-employed">Self-employed</option>
                    <option value="Transport: type 3">Transport: type 3</option>
                    <option value="Business Entity Type 3">Business Entity Type 3</option>
                    <option value="Government">Government</option>
                    <option value="Industry: type 9">Industry: type 9</option>
                    <option value="School">School</option>
                    <option value="Trade: type 2">Trade: type 2</option>
                    <option value="XNA">XNA</option>
                    <option value="Services">Services</option>
                    <option value="Bank">Bank</option>
                    <option value="Industry: type 3">Industry: type 3</option>
                    <option value="Other">Other</option>
                    <option value="Trade: type 6">Trade: type 6</option>
                    <option value="Industry: type 12">Industry: type 12</option>
                    <option value="Trade: type 7">Trade: type 7</option>
                    <option value="Postal">Postal</option>
                    <option value="Medicine">Medicine</option>
                    <option value="Housing">Housing</option>
                    <option value="Business Entity Type 2">Business Entity Type 2</option>
                    <option value="Construction">Construction</option>
                    <option value="Military">Military</option>
                    <option value="Industry: type 4">Industry: type 4</option>
                    <option value="Trade: type 3">Trade: type 3</option>
                    <option value="Legal Services">Legal Services</option>
                    <option value="Security">Security</option>
                    <option value="Industry: type 11">Industry: type 11</option>
                    <option value="University">University</option>
                    <option value="Business Entity Type 1">Business Entity Type 1</option>
                    <option value="Agriculture">Agriculture</option>
                    <option value="Security Ministries">Security Ministries</option>
                    <option value="Transport: type 2">Transport: type 2</option>
                    <option value="Industry: type 7">Industry: type 7</option>
                    <option value="Transport: type 4">Transport: type 4</option>
                    <option value="Telecom">Telecom</option>
                    <option value="Emergency">Emergency</option>
                    <option value="Police">Police</option>
                    <option value="Industry: type 1">Industry: type 1</option>
                    <option value="Transport: type 1">Transport: type 1</option>
                    <option value="Electricity">Electricity</option>
                    <option value="Industry: type 5">Industry: type 5</option>
                    <option value="Hotel">Hotel</option>
                    <option value="Restaurant">Restaurant</option>
                    <option value="Advertising">Advertising</option>
                    <option value="Mobile">Mobile</option>
                    <option value="Trade: type 1">Trade: type 1</option>
                    <option value="Industry: type 8">Industry: type 8</option>
                    <option value="Realtor">Realtor</option>
                    <option value="Cleaning">Cleaning</option>
                    <option value="Industry: type 2">Industry: type 2</option>
                    <option value="Trade: type 4">Trade: type 4</option>
                    <option value="Industry: type 6">Industry: type 6</option>
                    <option value="Culture">Culture</option>
                    <option value="Insurance">Insurance</option>
                    <option value="Religion">Religion</option>
                    <option value="Industry: type 13">Industry: type 13</option>
                    <option value="Industry: type 10">Industry: type 10</option>
                    <option value="Trade: type 5">Trade: type 5</option>
                </select><br>

                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """

# Prediction endpoint
@app.post("/evaluate", response_class=JSONResponse)
def predict_loan_status(
    DAYS_LAST_PHONE_CHANGE: float = Form(None),
    TOTALAREA_MODE: float = Form(None),
    WEEKDAY_APPR_PROCESS_START: str = Form(None),
    AMT_ANNUITY: float = Form(None),
    AMT_CREDIT: float = Form(None),
    EXT_SOURCE_2: float = Form(None),
    EXT_SOURCE_3: float = Form(None),
    OCCUPATION_TYPE: str = Form(None),
    ORGANIZATION_TYPE: str = Form(None)
):
    try:
        # Check if at least one value is provided
        if (DAYS_LAST_PHONE_CHANGE is None and TOTALAREA_MODE is None and WEEKDAY_APPR_PROCESS_START is None
            and AMT_ANNUITY is None and AMT_CREDIT is None and EXT_SOURCE_2 is None
            and EXT_SOURCE_3 is None and OCCUPATION_TYPE is None and ORGANIZATION_TYPE is None):
            raise HTTPException(status_code=400, detail="At least one field must be filled.")

        # Create a DataFrame with the input values
        data = {
            "DAYS_LAST_PHONE_CHANGE": [DAYS_LAST_PHONE_CHANGE],
            "TOTALAREA_MODE": [TOTALAREA_MODE],
            "TOTALAREA_MODE": [TOTALAREA_MODE],
            "WEEKDAY_APPR_PROCESS_START": [WEEKDAY_APPR_PROCESS_START],
            "AMT_ANNUITY": [AMT_ANNUITY],
            "AMT_CREDIT": [AMT_CREDIT],
            "EXT_SOURCE_2": [EXT_SOURCE_2],
            "EXT_SOURCE_3": [EXT_SOURCE_3],
            "OCCUPATION_TYPE": [OCCUPATION_TYPE],
            "ORGANIZATION_TYPE": [ORGANIZATION_TYPE]
        }

        # Fill missing values with some default value if necessary
        input_df = pd.DataFrame(data)

        # Make a prediction using the model
        prediction = default_classifier.predict(input_df)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
