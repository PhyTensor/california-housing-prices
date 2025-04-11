import requests
from nicegui import ui

ui.label('🏡 House Price Predictor').classes('text-2xl mb-6')

income = ui.input('Median Income').props('type=number step=0.01')
age = ui.input('Housing Median Age').props('type=number step=1')
rooms = ui.input('Rooms per Household').props('type=number step=0.01')
bedrooms = ui.input('Bedrooms per Room').props('type=number step=0.01')

proximity = ui.select(
    ['<1H OCEAN', 'INLAND'],
    label='Ocean Proximity'
).classes('w-64')

result = ui.label('').classes('mt-4 text-lg')

def predict():
    try:
        data = {
            "median_income": float(income.value),
            "housing_median_age": float(age.value),
            "room_per_household": float(rooms.value),
            "bedrooms_per_room": float(bedrooms.value),
            "ocean_proximity": proximity.value
        }
        response = requests.post("http://localhost:8000/predict", json=data)
        prediction = response.json().get("predicted_price", "N/A")
        result.set_text(f'💰 Predicted Price: ${prediction}')
    except Exception as e:
        result.set_text(f'❌ Error: {e}')

ui.button('Predict', on_click=predict).classes('mt-4 w-32')

ui.run()
