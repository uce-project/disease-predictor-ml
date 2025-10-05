import streamlit as st
import joblib
import pandas as pd
st.title("Disease Prediction Model")


with st.spinner("Please wait ✌️ Loading model..."):
    clf = joblib.load('model/random_forest_disease_model.pkl')
    le = joblib.load('model/label_encoder.pkl')
    mlb = joblib.load('model/symptom_binarizer.pkl')
    df = pd.read_csv('model/dataset/symptom_precaution.csv')
    df['Disease'] = df['Disease'].str.strip()


symptom_list = ['itching', 'skin_rash', 'continuous_sneezing', 'shivering', 'stomach_pain', 'acidity', 'vomiting', 'indigestion', 'muscle_wasting', 'patches_in_throat', 'fatigue', 'weight_loss', 'sunken_eyes', 'cough', 'headache', 'chest_pain', 'back_pain', 'weakness_in_limbs', 'chills', 'joint_pain', 'yellowish_skin', 'constipation', 'pain_during_bowel_movements', 'breathlessness', 'cramps', 'weight_gain', 'mood_swings', 'neck_pain', 'muscle_weakness', 'stiff_neck', 'pus_filled_pimples', 'burning_micturition', 'bladder_discomfort', 'high_fever', 'nodal_skin_eruptions', 'ulcers_on_tongue', 'loss_of_appetite', 'restlessness', 'dehydration', 'dizziness', 'weakness_of_one_body_side', 'lethargy', 'nausea', 'abdominal_pain', 'pain_in_anal_region', 'sweating', 'bruising', 'cold_hands_and_feets', 'anxiety', 'knee_pain', 'swelling_joints', 'blackheads', 'foul_smell_of urine', 'skin_peeling', 'blister', 'dischromic _patches', 'watering_from_eyes', 'extra_marital_contacts', 'diarrhoea', 'loss_of_balance', 'blurred_and_distorted_vision', 'altered_sensorium', 'dark_urine', 'swelling_of_stomach', 'bloody_stool', 'obesity', 'hip_joint_pain', 'movement_stiffness', 'spinning_movements', 'scurring', 'continuous_feel_of_urine', 'silver_like_dusting', 'red_sore_around_nose', 'nan', 'spotting_ urination', 'passage_of_gases', 'irregular_sugar_level', 'family_history', 'lack_of_concentration', 'excessive_hunger', 'yellowing_of_eyes', 'distention_of_abdomen', 'irritation_in_anus', 'swollen_legs', 'painful_walking', 'small_dents_in_nails', 'yellow_crust_ooze', 'internal_itching', 'mucoid_sputum', 'history_of_alcohol_consumption', 'swollen_blood_vessels', 'unsteadiness', 'inflammatory_nails', 'depression', 'fluid_overload', 'swelled_lymph_nodes', 'malaise', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 'fast_heart_rate', 'irritability', 'muscle_pain', 'mild_fever', 'yellow_urine', 'phlegm', 'enlarged_thyroid', 'increased_appetite', 'visual_disturbances', 'brittle_nails', 'drying_and_tingling_lips', 'polyuria', 'pain_behind_the_eyes', 'toxic_look_(typhos)', 'throat_irritation', 'swollen_extremeties', 'slurred_speech', 'red_spots_over_body', 'belly_pain', 'receiving_blood_transfusion', 'acute_liver_failure', 'redness_of_eyes', 'rusty_sputum', 'abnormal_menstruation', 'receiving_unsterile_injections', 'coma', 'sinus_pressure', 'palpitations', 'stomach_bleeding', 'runny_nose', 'congestion', 'blood_in_sputum', 'loss_of_smell']


def get_precautions(disease_name):
    row = df[df['Disease'] == disease_name]
    if row.empty:
        return None
    precautions = row[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()
    return [p for p in precautions if pd.notna(p)]


if "selected_symptoms_list" not in st.session_state:
    st.session_state.selected_symptoms_list = []



selected_symptom = st.selectbox("Select a symptom", [""] + symptom_list)


if selected_symptom and selected_symptom not in st.session_state.selected_symptoms_list:
    st.session_state.selected_symptoms_list.append(selected_symptom)


st.write("All selected symptoms so far:")


for symptom in st.session_state.selected_symptoms_list.copy():  
    col1, col2 = st.columns([0.8, 0.2])
    col1.write(symptom)
    if col2.button("❌", key=f"remove_{symptom}"):
        st.session_state.selected_symptoms_list.remove(symptom)
        st.rerun()  
if st.button("Predict", type="primary"):
    with st.spinner("Predicting..."):
        if st.session_state.selected_symptoms_list:
            print(st.session_state.selected_symptoms_list)
            x = mlb.transform([st.session_state.selected_symptoms_list])
            y = le.inverse_transform([clf.predict(x)[0]])[0]
            st.info(f"Predicted disease: {y}")
            precautions = get_precautions(y.strip())
            if precautions:
                st.subheader("Precautions:")
                for i, precaution in enumerate(precautions, 1):
                    st.write(f"{i}. {precaution}")

        else:
            st.warning("Please select at least one symptom before predicting!")
