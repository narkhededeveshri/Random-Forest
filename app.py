import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model from random.pkl
try:
    with open("random.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML Template with modern, attractive styling embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI & Workforce Analytics Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --border: #e5e7eb;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            background: var(--primary);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .header p {
            font-size: 0.95rem;
            opacity: 0.9;
        }

        .form-grid {
            padding: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }

        input, select {
            padding: 0.75rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 0.95rem;
            color: var(--text-main);
            background: #f9fafb;
            transition: all 0.2s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.875rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.1s ease;
            margin-top: 1rem;
        }

        .btn-submit:hover {
            background: var(--primary-hover);
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .result-card {
            margin: 0 2rem 2rem 2rem;
            padding: 1.5rem;
            background: #e0e7ff;
            border: 1px solid #c7d2fe;
            border-radius: 12px;
            text-align: center;
            color: #3730a3;
        }

        .result-card h3 {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .error-card {
            margin: 0 2rem 2rem 2rem;
            padding: 1rem;
            background: #fee2e2;
            border: 1px solid #fecaca;
            border-radius: 8px;
            color: #991b1b;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Workforce & AI Impact Predictor</h1>
            <p>Fill out the enterprise indicators below to run model prediction</p>
        </div>

        {% if error %}
        <div class="error-card" style="margin-top: 2rem;">
            <p>{{ error }}</p>
        </div>
        {% endif %}

        {% if prediction is not none %}
        <div class="result-card" style="margin-top: 2rem;">
            <h3>Prediction Result</h3>
            <p style="font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem;">{{ prediction }}</p>
        </div>
        {% endif %}

        <form action="/" method="POST" class="form-grid">
            <div class="form-group">
                <label for="company_name">Company Name (Encoded/ID)</label>
                <input type="number" step="any" name="company_name" required placeholder="e.g. 1">
            </div>
            <div class="form-group">
                <label for="industry">Industry Code</label>
                <input type="number" step="any" name="industry" required placeholder="e.g. 3">
            </div>
            <div class="form-group">
                <label for="country">Country Code</label>
                <input type="number" step="any" name="country" required placeholder="e.g. 2">
            </div>
            <div class="form-group">
                <label for="company_size">Company Size</label>
                <input type="number" step="any" name="company_size" required placeholder="e.g. 5000">
            </div>
            <div class="form-group">
                <label for="month">Month</label>
                <input type="number" min="1" max="12" name="month" required placeholder="e.g. 6">
            </div>
            <div class="form-group">
                <label for="year">Year</label>
                <input type="number" name="year" required placeholder="e.g. 2026">
            </div>
            <div class="form-group">
                <label for="layoffs_count">Layoffs Count</label>
                <input type="number" step="any" name="layoffs_count" required placeholder="e.g. 150">
            </div>
            <div class="form-group">
                <label for="layoff_percentage">Layoff Percentage</label>
                <input type="number" step="any" name="layoff_percentage" required placeholder="e.g. 5.5">
            </div>
            <div class="form-group">
                <label for="reason_for_layoffs">Reason for Layoffs Code</label>
                <input type="number" step="any" name="reason_for_layoffs" required placeholder="e.g. 1">
            </div>
            <div class="form-group">
                <label for="ai_automation_impact">AI Automation Impact</label>
                <input type="number" step="any" name="ai_automation_impact" required placeholder="e.g. 0.7">
            </div>
            <div class="form-group">
                <label for="ai_replacement_risk">AI Replacement Risk</label>
                <input type="number" step="any" name="ai_replacement_risk" required placeholder="e.g. 0.4">
            </div>
            <div class="form-group">
                <label for="open_roles">Open Roles</label>
                <input type="number" step="any" name="open_roles" required placeholder="e.g. 25">
            </div>
            <div class="form-group">
                <label for="hiring_trend">Hiring Trend Score</label>
                <input type="number" step="any" name="hiring_trend" required placeholder="e.g. 1.2">
            </div>
            <div class="form-group">
                <label for="remote_jobs_percentage">Remote Jobs %</label>
                <input type="number" step="any" name="remote_jobs_percentage" required placeholder="e.g. 40">
            </div>
            <div class="form-group">
                <label for="top_hiring_role">Top Hiring Role Code</label>
                <input type="number" step="any" name="top_hiring_role" required placeholder="e.g. 2">
            </div>
            <div class="form-group">
                <label for="stock_growth_percent">Stock Growth %</label>
                <input type="number" step="any" name="stock_growth_percent" required placeholder="e.g. 12.5">
            </div>
            <div class="form-group">
                <label for="revenue_growth_percent">Revenue Growth %</label>
                <input type="number" step="any" name="revenue_growth_percent" required placeholder="e.g. 8.0">
            </div>
            <div class="form-group">
                <label for="salary_budget_change">Salary Budget Change %</label>
                <input type="number" step="any" name="salary_budget_change" required placeholder="e.g. 3.0">
            </div>
            <div class="form-group">
                <label for="ai_adoption_level">AI Adoption Level</label>
                <input type="number" step="any" name="ai_adoption_level" required placeholder="e.g. 4">
            </div>
            <div class="form-group">
                <label for="employee_sentiment">Employee Sentiment Score</label>
                <input type="number" step="any" name="employee_sentiment" required placeholder="e.g. 3.5">
            </div>
            <div class="form-group">
                <label for="job_security_score">Job Security Score</label>
                <input type="number" step="any" name="job_security_score" required placeholder="e.g. 7.5">
            </div>

            <button type="submit" class="btn-submit">Calculate Prediction</button>
        </form>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    
    if request.method == "POST":
        if model is None:
            error = "Model could not be loaded. Please ensure 'random.pkl' is present."
        else:
            try:
                # Extract all 21 feature inputs matching model's expected feature list order
                feature_keys = [
                    "company_name", "industry", "country", "company_size", "month", "year",
                    "layoffs_count", "layoff_percentage", "reason_for_layoffs", "ai_automation_impact",
                    "ai_replacement_risk", "open_roles", "hiring_trend", "remote_jobs_percentage",
                    "top_hiring_role", "stock_growth_percent", "revenue_growth_percent",
                    "salary_budget_change", "ai_adoption_level", "employee_sentiment", "job_security_score"
                ]
                
                features = [float(request.form[key]) for key in feature_keys]
                final_features = np.array([features])
                
                # Perform prediction
                pred = model.predict(final_features)
                prediction = str(pred[0])
            except Exception as e:
                error = f"Error during prediction processing: {str(e)}"
                
    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
