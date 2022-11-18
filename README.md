# NeurIPS_TrojanDetectionChallenge
Evasive Trojans Track: the task is to create Trojaned networks that are hard to detect.
# Defender: 4 different metrics
- ASR: Submissions will first be tested to see whether the attack specifications are satisfied. Namely, the average attack success rate with respect to the provided attack specifications must be at least 97% for the submitted Trojaned models.
- Accuracy-based detector: This detector assumes that Trojaned models will reliably have lower accuracy than clean models.
- Specificity-based detector: This detector assumes that Trojaned models trained with a particular trigger will also be susceptible to other triggers from the same distribution. We assume that the defender has full knowledge of the trigger distribution and can sample from it. The network under examination is given a set of inputs with randomly sampled Trojan triggers, and entropy of the average posterior is recorded.
- MNTD: This is a simple meta-network detector based on the well-known MNTD Trojan detection method from the literature (Usenix Security). This detector also serves as a baseline for the Trojan Detection Track. The detection score is the logit, or equivalently the predicted probability of a model being Trojaned. Due to the limited dataset size, the detection scores are obtained via 5-fold cross-validation. That is, we train 5 MNTD models on different splits of the held-out clean models and submitted Trojaned models. This gives an AUROC score for MNTD. To reduce variance, we repeat this procedure three times and average the results to obtain a final AUROC for MNTD.
# Attacker: 200 trojan MNIST models
Adopted an adaptive attack strategy to bypass all these detection schemes while keeping the attack success rate. Our main idea is step by step optimization, one by one attack. Firstly, we trained 200 models with higher ASR and lower MNTD AUC. Then, we fine-tuned these models to improve their ACC. Finally, we selected some specificity models to reduce the detection effect. Our team gained the 2nd place in this competition.
# How to run
## Defender
Run defense/VisualTrojan.py, please note you should rename all paths.
## Attacker
Please change the clean model and attack specificity path to your path before starting to run.
- Step 1: Train 200 trojan models by fine-tuning on 200 clean models that have similar outputs to clean models when predicting non-trigger samples, making MNTD detection ineffective. Run train_batch_of_model.py in the "step1" folder and it will save the model to the "temp1" folder.
- Step 2: Reduce the learning rate to fine-tune the model in "temp1", but the other settings remain the same as in step 1. This stage saves the model with the best accuracy, making ACC testing ineffective. Run train_batch_of_model.py in the "step2" folder and it will save the model to the "temp2" folder.
- Step 3: Use specificity detection to filter the indexes of the less specific part of the model in the "temp2" folder. By default, 20 indexes are filtered but can be adjusted according to the actual situation. Run VisualTrojan.py in the "step3" folder to screen out the models that match the requirements and it copies the rest of the models to the "result" folder.
- Step 4: Fine-tune part of the model in "temp2" according to the index in Step 3, which can optimize the specificity score to avoid specificity detection. Run train_batch_of_model.py in the "step4" folder to fine-tune the indexed model and it will save the model to the "result" folder.
After running the four steps, the "result" folder contains the final 200 trojan models.
