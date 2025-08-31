# AI Poker Bot – Defence Cyber Marvel Challenge

## Overview
This project was developed for the **Defence Cyber Marvel 4 Games Night Challenge**, where the objective was to design an advanced **AI Poker Bot** capable of competing in **heads-up Texas Hold’em**.

The bot integrates **probability, game theory, and machine learning** to achieve a balanced style of play that combines **Game Theory Optimal (GTO) strategies** with **exploitative adaptations** against opponents.

**Core Objectives:**
- Build a **robust pre-flop and flop strategy** grounded in combinatorial hand evaluation and board texture analysis.
- Implement **data preprocessing and feature engineering** for hand classification.
- Train and evaluate **ML models** to predict hand strength and optimal actions.
- Incorporate **opponent modelling** to adjust strategies dynamically.

---

##  Features

### 1. Preprocessing Pipeline (`preprocess.py`)
Transforms raw poker hand datasets into structured features suitable for ML models.

- **Data Input:**
  - `poker-hand-training-true.data`
  - `poker-hand-testing.data`
  - `poker10m` (10 million hand dataset for large-scale analysis)

- **Feature Engineering:**
  - Card sorting for consistency
  - Rank patterns (pairs, trips, quads, straights)
  - Suit patterns (flush detection & distribution)
  - Strength features (straight, flush, straight flush, royal flush)
  - Advanced features (rank entropy & suit entropy)

- **Output:**
  - Processed dataset saved as `X_train_pre.csv` under `/data/processed/`

---

### 2. Pre-Flop Strategy
The pre-flop phase provides the foundation of the bot’s strategy before community cards are revealed.

- **Hand Evaluation**
  - Categorises hole cards into *Premium, Strong, Speculative, Marginal*
  - Uses the **Chen Formula** for numerical scoring

- **Positional Adjustments**
  - Early Position (tight ranges)
  - Middle Position (balanced)
  - Late Position (wide, aggressive)

- **Stack Depth & Risk**
  - **Stack-to-Pot Ratio (SPR)** for aggression scaling
  - Tournament adjustments with **Independent Chip Model (ICM)**

- **Decision Engine**
  - Dynamically chooses *fold, call, raise* based on hand, position, and SPR

---

### 3. Flop Strategy
Once community cards are revealed, the bot incorporates **board texture, equity evaluation, and opponent modelling**.

- **Board Texture Analysis**
  - Dry boards → small continuation bets
  - Wet boards → larger bets, reduced bluffs
  - Paired/polarised → adjusted aggression

- **Hand Equity & Bayesian Updates**
  - Real-time probability updates refine winning chances
  - Bluff frequency adjusted dynamically

- **Position-Based Play**
  - *In Position (IP)* → more frequent continuation bets
  - *Out of Position (OOP)* → controlled aggression, check-raises

- **Opponent Modelling**
  - Tracks metrics: VPIP, PFR, Aggression Factor, C-Bet %, etc.
  - Categorises into **TAG, LAG, NIT, Calling Station**
  - Exploits weaknesses with tailored bet sizing and bluffing strategies

### 4. 🤖 Machine Learning Integration
Several ML models were tested for hand classification and decision support:

- **Models Tested:** Decision Tree, XGBoost, Random Forest, LSTM  
- **Final Model:** **Random Forest**  
  - Best trade-off between accuracy, robustness, and interpretability  
  - Outputs probability distributions for hand strength rather than fixed rules  

**Integration into Decision Engine:**
- Bet sizing scaled with predicted Expected Value (EV)  
- Bluff frequency increased if opponent over-folds  
- Check-raise strategies deployed vs. high aggression profiles  

## System Architecture

**Modules:**
- **Hand Analyzer** – evaluates pre-flop hand strength
- **Board Analyzer** – classifies flop textures & nut potential
- **Stack & Risk Manager** – incorporates SPR and ICM
- **Opponent Model** – builds statistical profiles of opponents
- **Decision Engine** – integrates data to select optimal action

**Flow:**


## Example Workflow

1. **Input**: Hole cards + game state  
2. **Pre-Flop**: Hand scored using Chen Formula + positional weighting  
3. **Flop**: Board texture analysed, hand equity updated  
4. **Opponent Model**: Adjusts strategy based on tendencies  
5. **Decision Engine**: Outputs action → *Fold, Call, Raise, Bluff, Trap*  

##  Future Improvements

- **Reinforcement Learning**: Deep Q-Learning & Monte Carlo Tree Search for self-play optimisation  
- **Multi-Table Support**: Expand to 6-max and 9-max play  
- **Real-Time Adaptation**: Bayesian updates during live play  
- **Cross-Domain Applications**:
  - Algorithmic trading
  - Cyber defence modelling
  - Military decision simulations  




