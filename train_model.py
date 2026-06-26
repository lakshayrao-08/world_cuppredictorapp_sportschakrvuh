import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'world_cup_model.pkl')

df = pd.read_csv(DATA_PATH)

features_list = []
targets_list = []

versions = df['version'].unique()

for v in versions:
    df_v = df[df['version'] == v]
    teams_v = df_v['team'].tolist()
    
    for i in range(len(teams_v)):
        for j in range(i + 1, len(teams_v)):
            teamA_stats = df_v[df_v['team'] == teams_v[i]].iloc[0]
            teamB_stats = df_v[df_v['team'] == teams_v[j]].iloc[0]
            
            rank_diff = teamA_stats['fifa_rank_pre_tournament'] - teamB_stats['fifa_rank_pre_tournament']
            val_diff = teamA_stats['squad_total_market_value_eur'] - teamB_stats['squad_total_market_value_eur']
            goals_diff = teamA_stats['goals_scored_last_4y'] - teamB_stats['goals_scored_last_4y']
            
            features_list.append([rank_diff, val_diff, goals_diff])
            
            if teamA_stats['winner'] == 1:
                targets_list.append(2)
            elif teamB_stats['winner'] == 1:
                targets_list.append(0)
            else:
                targets_list.append(1)

X = pd.DataFrame(features_list, columns=['rank_diff', 'market_value_diff', 'goals_trend_diff'])
y = np.array(targets_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

original_predict_proba = model.predict_proba

def bounded_predict_proba(X_input):
    raw_probs = original_predict_proba(X_input)
    adjusted_probs = []
    
    for prob in raw_probs:
        if len(prob) == 3:
            p_b, p_d, p_a = prob[0], prob[1], prob[2]
        else:
            p_b, p_d, p_a = prob[0], 0.20, 1.0 - prob[0] - 0.20
            
        rank_d = X_input.iloc[0]['rank_diff'] if hasattr(X_input, 'iloc') else X_input[0][0]
        
        min_limit = 0.30 if rank_d < -20 else 0.20
        max_limit = 0.60
        
        p_a = max(min_limit, min(max_limit, p_a))
        p_b = max(min_limit, min(max_limit, p_b))
        
        p_d = 1.0 - p_a - p_b
        
        if p_d < 0.10:
            diff = 0.10 - p_d
            p_d = 0.10
            if p_a > p_b:
                p_a -= diff
            else:
                p_b -= diff
                
        adjusted_probs.append([p_b, p_d, p_a])
        
    return np.array(adjusted_probs)

model.predict_proba = bounded_predict_proba

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("Advanced Random Forest Engine Trained & Exported Successfully!")