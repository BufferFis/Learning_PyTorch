import torch.nn as nn
import torch

def NEVERCALL():
    mse_loss = nn.MSELoss() # No outliers
    mae_loss = nn.L1Loss() # outliers fine
    huber_loss = nn.SmoothL1Loss() # Some outliers

    ce_loss = nn.CrossEntropyLoss()
    bce_logits_loss = nn.BCEWithLogitsLoss() # Binary classification but better than sigmoid + BCE

    bce_loss = nn.BCELoss()

    # START HERE: What type of problem?

    if problem_type == "regression":
        if data_has_many_outliers:
            loss = nn.L1Loss()          # Robust to outliers
        elif data_has_some_outliers:
            loss = nn.SmoothL1Loss()    # Best of both worlds
        else:
            loss = nn.MSELoss()         # Standard choice
            
    elif problem_type == "classification":
        if num_classes == 2:
            loss = nn.BCEWithLogitsLoss()  # Binary: spam/not spam
        elif num_classes > 2 and exclusive_classes:
            loss = nn.CrossEntropyLoss()   # Multi-class: cat XOR dog XOR bird
        elif num_classes > 2 and non_exclusive_classes:
            loss = nn.BCELoss()            # Multi-label: person AND car AND outdoor
# House price prediction → MSELoss
# Unless you have crazy mansion outliers → L1Loss

# Disease diagnosis (yes/no) → BCEWithLogitsLoss
# Disease type (cancer/diabetes/healthy) → CrossEntropyLoss  
# Symptom detection (multiple symptoms) → BCELoss
# Blood pressure prediction → MSELoss (unless sensor noise → L1Loss)

# Post sentiment (positive/negative/neutral) → CrossEntropyLoss
# Content tags (funny, sad, inspiring) → BCELoss
# Engagement prediction (likes count) → MSELoss
# Spam detection → BCEWithLogitsLoss

# Object classification (car/person/bike) → CrossEntropyLoss
# Bounding box regression → SmoothL1Loss (handles outliers)
# Distance estimation → L1Loss (sensor noise)
# Lane detection (multiple lanes) → BCELoss

# Stock price prediction → L1Loss (market crashes = outliers)
# Credit approval → BCEWithLogitsLoss
# Risk category (low/medium/high) → CrossEntropyLoss
# Portfolio return → SmoothL1Loss (some volatility expected)



"""TIPS"""

# Start slow
# Regression → MSELoss
# Binary Classification → BCEWithLogitsLoss  
# Multi-class → CrossEntropyLoss
# Multi-label → BCELoss

# switch up
# Training unstable? → Try L1Loss instead of MSELoss
# Poor performance on outliers? → SmoothL1Loss
# Numerical instability? → BCEWithLogitsLoss instead of Sigmoid + BCELoss


"""Custom LOss"""

def custom_loss(prediction, target):
    mse = nn.MSELoss()(prediction, target)
    l1 = nn.L1Loss()(prediction, target)
    return 0.7 * mse + 0.3 * l1

def regularized(predictions, target):
    base = nn.CrossEntropyLoss()(predictions, target)
    l2_reg = sum(p.pow(2).sum() for p in model.parameters())
    return base_loss + 0.001 * l2_reg


    