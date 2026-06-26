from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'world_cup_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'train.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            current_model = pickle.load(f)
    else:
        return jsonify({'error': 'Backend binary configuration missing.'}), 500

    if os.path.exists(DATA_PATH):
        df_data = pd.read_csv(DATA_PATH)
        latest_version = df_data['version'].max()
        df_latest = df_data[df_data['version'] == latest_version]
    else:
        return jsonify({'error': 'Core dataset structure unavailable.'}), 500
        
    req = request.get_json()
    teamA = req.get('teamA')
    teamB = req.get('teamB')
    
    try:
        statsA = df_latest[df_latest['team'] == teamA].iloc[0]
        statsB = df_latest[df_latest['team'] == teamB].iloc[0]
    except IndexError:
        return jsonify({'error': f'Database lookup failed for {teamA} or {teamB}. Check exact spelling in train.csv.'}), 400
        
    rankA = int(statsA['fifa_rank_pre_tournament'])
    rankB = int(statsB['fifa_rank_pre_tournament'])
    
    rank_diff = rankA - rankB
    val_diff = statsA['squad_total_market_value_eur'] - statsB['squad_total_market_value_eur']
    goals_diff = statsA['goals_scored_last_4y'] - statsB['goals_scored_last_4y']
    
    input_payload = pd.DataFrame([[rank_diff, val_diff, goals_diff]], columns=['rank_diff', 'market_value_diff', 'goals_trend_diff'])
    
    probabilities = current_model.predict_proba(input_payload)[0]
    
    if len(probabilities) == 3:
        p_b, p_d, p_a = probabilities[0], probabilities[1], probabilities[2]
    else:
        p_b, p_d, p_a = probabilities[0], 0.25, 1.0 - probabilities[0] - 0.25

    if rankA <= 20 and rankB <= 20 and abs(rank_diff) <= 15:
        min_limit, max_limit = 0.40, 0.50
    else:
        min_limit, max_limit = 0.30, 0.60

    p_a = max(min_limit, min(max_limit, p_a))
    p_b = max(min_limit, min(max_limit, p_b))
    
    p_d = 1.0 - p_a - p_b
    
    if p_d < 0.15:
        diff = 0.15 - p_d
        p_d = 0.15
        if p_a > p_b:
            p_a -= diff
        else:
            p_b -= diff

    prob_win_a = round(p_a * 100, 1)
    prob_win_b = round(p_b * 100, 1)
    prob_draw = round(p_d * 100, 1)
    
    valA_m = round(statsA['squad_total_market_value_eur'] / 1000000, 1)
    valB_m = round(statsB['squad_total_market_value_eur'] / 1000000, 1)
    financial_gap = abs(valA_m - valB_m)
    
    feature_importances = current_model.feature_importances_
    importance_rank = round(feature_importances[0] * 100, 1)
    importance_value = round(feature_importances[1] * 100, 1)
    importance_goals = round(feature_importances[2] * 100, 1)
    
    goalsA = int(statsA['goals_scored_last_4y'])
    goalsB = int(statsB['goals_scored_last_4y'])
    
    if prob_win_a > prob_win_b and prob_win_a > prob_draw:
        insight = f"The core analytical metrics favor {teamA}. Higher squad valuation and superior tournament consistency index parameters back this simulation outcome."
    elif prob_win_b > prob_win_a and prob_win_b > prob_draw:
        insight = f"Simulation indicates an upper hand for {teamB}. Their recent attacking efficiency metrics heavily outpace defensive containment systems."
    else:
        insight = "Critical equilibrium state detected between both configurations. Anticipate a high-intensity tactical draw leading straight to penalty shootouts."
        
    return jsonify({
        'winA': prob_win_a,
        'draw': prob_draw,
        'winB': prob_win_b,
        'insight': insight,
        'analytics': {
            'teamA_val': f"{valA_m}M EUR",
            'teamB_val': f"{valB_m}M EUR",
            'financial_gap': f"{financial_gap}M EUR",
            'importance_rank': importance_rank,
            'importance_value': importance_value,
            'importance_goals': importance_goals,
            'goalsA': goalsA,
            'goalsB': goalsB,
            'rankA': rankA,
            'rankB': rankB
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)