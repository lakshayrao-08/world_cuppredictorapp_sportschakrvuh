
A predictive simulation system built with Python, Flask, and Random Forest Classifier to analyze and forecast tournament match outcomes based on FIFA ranks, squad values, and recent goal trends.
# Tournament Predictive Simulation System

An advanced data analytics and predictive modeling platform designed to simulate tournament match outcomes. The system uses a machine learning engine built with a Random Forest Classifier to process historical performance statistics, squad valuations, and competitive parameters to deliver balanced probability forecasts.

## Key Features

* **ML Engine**: Trained on performance datasets utilizing multi-class probability outputs.
* **Balanced Probabilities**: Integrated algorithmic caps preventing skewed predictions, maintaining premium competitive matches within strict brackets (40% - 50%).
* **Real-time Analytics**: Computes financial gaps, squad value significance, and historical scoring trend weights dynamically.
* **Asynchronous Interface**: Modern dashboard that connects seamlessly with a local Flask backend service.

## Tech Stack

* **Backend**: Python, Flask, Flask-CORS
* **Data & ML**: Pandas, NumPy, Scikit-Learn, Pickle
* **Frontend**: HTML5, CSS3, JavaScript (Fetch API)
