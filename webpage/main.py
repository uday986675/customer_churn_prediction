import gradio as gr
import numpy as np
import joblib

# Load your trained model (adjust the path and loader as needed)
model = joblib.load('/home/uday/Documents/projects/deployment/customer_churn_prediction/customer_churn_prediction.pkl')


# Define your features and possible categorical options
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# Categorical options (replace with your training data's categories)
gender_options = ["Female", "Male"]
binary_options = ["No", "Yes"]
multiple_lines_options = ["No phone service", "No", "Yes"]
internet_service_options = ["DSL", "Fiber optic", "No"]
online_security_options = binary_options + ["No internet service"]
online_backup_options = binary_options + ["No internet service"]
device_protection_options = binary_options + ["No internet service"]
tech_support_options = binary_options + ["No internet service"]
streaming_options = binary_options + ["No internet service"]
contract_options = ["Month-to-month", "One year", "Two year"]
paperless_billing_options = binary_options
payment_method_options = [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
]

# Example encoding dictionaries based on typical label encoding
# Replace these with your actual encodings or load them from saved encoders

gender_mapping = {"Female": 0, "Male":1}
binary_mapping = {"No":0, "Yes":1}
multiple_lines_mapping = {"No phone service":0, "No":1, "Yes":2}
internet_service_mapping = {"DSL":0, "Fiber optic":1, "No":2}
contract_mapping = {"Month-to-month":0, "One year":1, "Two year":2}
paperless_billing_mapping = {"No":0, "Yes":1}
payment_method_mapping = {
    "Electronic check":0,
    "Mailed check":1,
    "Bank transfer (automatic)":2,
    "Credit card (automatic)":3 
}

# Similar mappings should be created for all categorical variables
# For simplicity, the ones below use binary_mapping plus an extra category "No internet service" encoded as 2
extended_binary_mapping = {"No":0, "Yes":1, "No internet service":2}

def preprocess_inputs(
    gender, senior_citizen, partner, dependents,
    tenure, phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract, paperless_billing,
    payment_method, monthly_charges, total_charges
):
    # Encode categorical variables
    gender_enc = gender_mapping[gender]
    partner_enc = binary_mapping[partner]
    dependents_enc = binary_mapping[dependents]
    phone_service_enc = binary_mapping[phone_service]
    multiple_lines_enc = multiple_lines_mapping[multiple_lines]
    internet_service_enc = internet_service_mapping[internet_service]
    online_security_enc = extended_binary_mapping[online_security]
    online_backup_enc = extended_binary_mapping[online_backup]
    device_protection_enc = extended_binary_mapping[device_protection]
    tech_support_enc = extended_binary_mapping[tech_support]
    streaming_tv_enc = extended_binary_mapping[streaming_tv]
    streaming_movies_enc = extended_binary_mapping[streaming_movies]
    contract_enc = contract_mapping[contract]
    paperless_billing_enc = paperless_billing_mapping[paperless_billing]
    payment_method_enc = payment_method_mapping[payment_method]

    # Senior Citizen is usually 0 or 1 already, ensure int
    senior_citizen_enc = int(senior_citizen)

    # All other numeric fields
    tenure_val = float(tenure)
    monthly_charges_val = float(monthly_charges)
    total_charges_val = float(total_charges)

    # Construct feature array in correct order expected by your model
    input_vector = np.array([[
        gender_enc,
        senior_citizen_enc,
        partner_enc,
        dependents_enc,
        tenure_val,
        phone_service_enc,
        multiple_lines_enc,
        internet_service_enc,
        online_security_enc,
        online_backup_enc,
        device_protection_enc,
        tech_support_enc,
        streaming_tv_enc,
        streaming_movies_enc,
        contract_enc,
        paperless_billing_enc,
        payment_method_enc,
        monthly_charges_val,
        total_charges_val
    ]])

    return input_vector

def predict_churn(*args):
    try:
        input_vector = preprocess_inputs(*args)
        # Print shape debug (comment out in production)
        print(f"Input vector shape: {input_vector.shape}")
        prediction = model.predict(input_vector)

        # If model outputs probabilities, you might use predict_proba or threshold
        # For classification label predictions it might be 0/1 directly

        # Example for classifier returning labels:
        return "Churn" if prediction[0] == 1 else "No Churn"

    except Exception as e:
        # Return error string to Gradio interface
        return f"Error during prediction: {str(e)}"


# Build the Gradio Interface Inputs
inputs = [
    gr.Dropdown(choices=gender_options, label="Gender"),
    gr.Number(label="Senior Citizen (0 or 1)"),
    gr.Dropdown(choices=binary_options, label="Partner"),
    gr.Dropdown(choices=binary_options, label="Dependents"),
    gr.Number(label="Tenure"),
    gr.Dropdown(choices=binary_options, label="Phone Service"),
    gr.Dropdown(choices=multiple_lines_options, label="Multiple Lines"),
    gr.Dropdown(choices=internet_service_options, label="Internet Service"),
    gr.Dropdown(choices=extended_binary_mapping.keys(), label="Online Security"),
    gr.Dropdown(choices=extended_binary_mapping.keys(), label="Online Backup"),
    gr.Dropdown(choices=extended_binary_mapping.keys(), label="Device Protection"),
    gr.Dropdown(choices=extended_binary_mapping.keys(), label="Tech Support"),
    gr.Dropdown(choices=extended_binary_mapping.keys(), label="Streaming TV"),
    gr.Dropdown(choices=extended_binary_mapping.keys(), label="Streaming Movies"),
    gr.Dropdown(choices=contract_options, label="Contract"),
    gr.Dropdown(choices=binary_options, label="Paperless Billing"),
    gr.Dropdown(choices=payment_method_options, label="Payment Method"),
    gr.Number(label="Monthly Charges"),
    gr.Number(label="Total Charges"),
]

output = gr.Label(label="Prediction")

iface = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs=output,
    title="Customer Churn Prediction",
    description="Enter customer details to predict whether the customer will churn."
)

iface.launch(share=True)
