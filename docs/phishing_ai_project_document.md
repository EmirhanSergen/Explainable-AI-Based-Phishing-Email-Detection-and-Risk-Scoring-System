# Explainable AI-Based Phishing Email Detection and Risk Scoring System

## 1. Introduction
Phishing is one of the most widespread and damaging forms of cyber attack in modern digital communication. In a phishing attack, an adversary attempts to deceive users through fraudulent e-mails that appear trustworthy, urgent, or institutionally legitimate. The main purpose of such messages is usually to steal credentials, financial information, personal data, or to redirect users to malicious websites. In some cases, phishing e-mails are also used as an entry point for malware delivery, session hijacking, or broader social engineering campaigns.

E-mail remains one of the most commonly used communication channels in both personal and organizational environments. Because of this, e-mail security is still a critical component of information security. Even though many mail systems already include spam filters or blacklisting approaches, these traditional mechanisms often suffer from important limitations. They may classify messages as suspicious without explaining the reason, they may fail when the attacker slightly changes the wording of the message, and they usually provide a simple binary output instead of a risk-oriented interpretation.

This project aims to design and implement a lightweight but meaningful Artificial Intelligence-based phishing detection system. The proposed system will analyze e-mail text, classify whether the message is phishing or legitimate, estimate the level of security risk, and provide explainable insights about the factors that influenced the model decision. In this way, the project goes beyond a basic classification task and becomes a small security decision-support system.

---

## 2. Motivation and Problem Definition
The main problem addressed in this project is that phishing e-mails are increasingly sophisticated, persuasive, and difficult for ordinary users to distinguish from legitimate communications. Attackers often use language such as urgency, authority, fear, account verification warnings, password reset requests, payment notifications, or security alerts to manipulate the target into immediate action.

From a security perspective, phishing is not only a messaging problem but also a gateway to more severe security incidents. A successful phishing attack may result in:
- unauthorized access to user accounts,
- theft of authentication credentials,
- disclosure of confidential information,
- installation of malicious software,
- compromise of organizational systems,
- financial loss,
- and reputational damage.

Traditional detection systems can be useful, but many of them work as closed black-box filters. They may block or allow messages without providing an interpretable explanation. In educational and organizational settings, however, understanding *why* a message is risky is also valuable. Therefore, the project addresses not only detection performance but also interpretability and practical security relevance.

The core problem can be stated as follows:

**How can a lightweight machine learning system be used to detect phishing e-mails, estimate their severity as a security risk, and explain the basis of its decision in an understandable manner?**

---

## 3. Project Aim and Objectives
The overall aim of the project is to build an explainable and risk-aware phishing e-mail analysis pipeline using classical machine learning methods.

### Main objectives
The project has the following main objectives:

1. **To detect phishing e-mails automatically** using textual features extracted from e-mail content.
2. **To compare multiple machine learning models** and evaluate which one performs best for this problem.
3. **To assign a risk level** to each analyzed e-mail rather than providing only a binary label.
4. **To explain model decisions** using Explainable AI techniques so that suspicious indicators can be interpreted by the user.
5. **To relate the implementation to information security concepts** such as e-mail security, social engineering, authentication threats, and risk management.

### Secondary objectives
If time and project scope allow, the following secondary objectives may also be considered:
- testing the system on a larger dataset,
- analyzing the effect of suspicious URLs and security-related keywords,
- and demonstrating how small wording changes can affect detection reliability.

---

## 4. Course Relevance
This project is strongly aligned with the content of the CENG374 Introduction to Computer Security course. It is not only a machine learning exercise but also a security-oriented implementation task.

### Direct relation to course topics
- **Week 3 – Information Security Standards and Risk Management:**
  The project includes a probability-based and rule-assisted risk scoring layer. This allows the system to provide a risk-oriented decision rather than a simple label.

- **Week 4 – Classification of Threats and Types of Attacks:**
  Phishing is a common social engineering and cyber attack method. The project directly focuses on detecting this type of threat.

- **Week 9 – User Authentication and Access Control:**
  Many phishing attacks aim to steal credentials and bypass authentication security. Therefore, phishing is directly connected to authentication-related threats.

- **Week 13 – Attack Detection:**
  The classification pipeline can be interpreted as a form of attack detection applied to e-mail content.

- **Week 14 – Web and Email Security:**
  The project is especially relevant here because it focuses on malicious e-mail content and suspicious links.

### Broader relevance
The project also indirectly relates to:
- software security, because insecure systems are often exploited after phishing success,
- malware, because phishing e-mails may deliver malicious attachments or links,
- and information security fundamentals, because phishing threatens confidentiality, integrity, and sometimes availability.

---

## 5. Conceptual Security Background
To make the project academically grounded, the phishing problem should also be interpreted within the framework of computer security principles.

### 5.1 Phishing as a Social Engineering Attack
Phishing is a human-focused cyber attack that exploits trust, fear, curiosity, or urgency. Instead of attacking only software, it attacks the judgment of the user. For example, attackers may impersonate a bank, university, online service, or system administrator and pressure the target into clicking a link or sharing sensitive information.

### 5.2 Relation to the CIA Triad
The project can be explained using the CIA triad:
- **Confidentiality:** stolen passwords and personal information violate confidentiality.
- **Integrity:** attackers may manipulate account settings or transactional data after compromise.
- **Availability:** some phishing campaigns may lead to account lockouts, service abuse, or operational disruption.

### 5.3 Authentication and Identity Theft
One of the most common goals of phishing is to capture usernames, passwords, one-time codes, or session-related data. Therefore, phishing directly threatens authentication mechanisms and user identity protection.

### 5.4 Risk Management Perspective
Not every suspicious e-mail has the same severity. Some messages may contain a weak phishing signal, while others may show multiple high-risk indicators such as urgent language, account verification prompts, password requests, and suspicious URLs. Because of this, adding a risk scoring layer makes the project more consistent with information security risk management principles.

---

## 6. Project Scope
The scope of the project is intentionally designed to be meaningful but manageable. The goal is not to build a production-grade e-mail gateway, but rather to develop an academic prototype that demonstrates phishing detection, risk analysis, and explainability in a practical way.

### Included in the project scope
The project includes:
- loading and analyzing a labeled phishing e-mail dataset,
- preprocessing e-mail text,
- extracting text-based and security-oriented features,
- training and comparing multiple machine learning models,
- generating risk scores,
- applying explainability methods,
- and demonstrating the system on sample e-mails.

### Excluded from the project scope
The project does not aim to include:
- live e-mail server integration,
- production-grade mail filtering,
- attachment malware scanning,
- real-time deployment across enterprise infrastructure,
- full URL reputation lookup services,
- or large-scale deep learning training with heavy transformer models.

Keeping these out of scope ensures that the project remains focused, feasible, and suitable for the available time.

---

## 7. Proposed System Overview
The system will take an e-mail text as input and produce four main outputs:

1. **Prediction:** Whether the e-mail is phishing or legitimate.
2. **Probability Score:** The model confidence for the phishing class.
3. **Risk Level:** A human-readable risk label such as Low, Medium, High, or Critical.
4. **Explanation:** A summary of which features, words, or indicators most influenced the decision.

This structure makes the project stronger than a conventional binary classifier. Instead of only saying “this is phishing,” the system will also indicate how risky the message is and why the model reached that conclusion.

---

## 8. Dataset Strategy
The initial dataset planned for the project is the phishing e-mail dataset hosted on Hugging Face:

**Dataset:** `zefang-liu/phishing-email-dataset`

This dataset is suitable for the first phase because it is relatively manageable in size, directly related to phishing e-mail detection, and convenient to access through Python data-loading tools.

### Why start with this dataset?
This dataset is preferred initially because:
- it is directly aligned with the project task,
- it contains labeled text samples,
- it is easier to preprocess than mixed-format datasets,
- and it enables quick prototyping.

### Possible second-phase dataset
If enough time remains after the first working version is completed, a larger or more complex phishing-related dataset may be tested in order to observe scalability, robustness, or possible performance improvements.

### Dataset-related tasks
The following steps will be performed during dataset preparation:
- inspect data columns and labels,
- analyze class balance,
- identify missing values,
- remove duplicates if necessary,
- and determine whether the text should be combined from subject and body fields if such fields exist.

---

## 9. Feature Engineering Approach
A strong part of the project is that it will not rely only on raw text classification. It will also integrate simple security-oriented indicators.

### 9.1 Text-Based Features
The main textual representation will be generated using **TF-IDF (Term Frequency–Inverse Document Frequency)**. TF-IDF is appropriate for this project because it is lightweight, interpretable, and highly effective for many classical text classification tasks.

Possible TF-IDF configuration may include:
- unigram and bigram extraction,
- removal of very rare terms,
- limiting vocabulary size for efficiency,
- and standard text normalization.

### 9.2 Security-Oriented Auxiliary Features
In addition to text representation, some handcrafted indicators may be extracted to reflect phishing behavior more explicitly. These may include:
- whether the e-mail contains a URL,
- the number of URLs,
- whether suspicious words such as “verify,” “password,” “login,” “urgent,” or “click here” appear,
- whether account-related or credential-related terms are present,
- and whether urgency expressions are used.

These indicators are useful because phishing messages often follow recognizable behavioral patterns. By combining text features with security-related heuristics, the project becomes more security-aware and better grounded in the domain.

---

## 10. Machine Learning Models
Instead of using only one model, the project will compare multiple machine learning algorithms in a gradual progression. This allows the report to present not only final results but also model evolution and comparison.

### Planned model set
The current plan is to evaluate three classical models:

1. **Naive Bayes**
   - Serves as a simple probabilistic baseline for text classification.
   - Often performs reasonably well on word-frequency-based tasks.
   - Useful for showing the starting point of the project.

2. **Logistic Regression**
   - A strong and widely used model for text classification.
   - Fast to train and easy to interpret.
   - Usually performs better than very simple baselines.

3. **Random Forest**
   - A tree-based model that may capture nonlinear interactions among features.
   - Useful as a comparison against linear models.
   - May perform differently when auxiliary security indicators are included.

### Why compare multiple models?
The comparison is important for several reasons:
- it demonstrates a proper experimental methodology,
- it helps justify model selection,
- and it shows that the project is not limited to a single arbitrary algorithm.

Rather than choosing a complex deep learning model, this project intentionally focuses on classical machine learning methods because they are lighter, more practical for the available time, and often sufficient for a meaningful academic prototype.

---

## 11. Model Evaluation Strategy
A phishing detection system should not be judged only by overall accuracy. In security problems, some errors are more dangerous than others.

### Evaluation metrics
The following evaluation metrics will be used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### Why Recall is especially important
In this project, recall is one of the most critical metrics because a false negative means a phishing e-mail was incorrectly classified as legitimate. In real life, this is a dangerous outcome because the malicious message may reach the user without warning.

### Importance of Precision as well
Precision is also important because a system that labels too many legitimate e-mails as phishing may become impractical or annoying for users. Therefore, the project should discuss the balance between detecting as many attacks as possible and minimizing unnecessary alarms.

### Experimental process
The dataset will be divided into training and testing subsets, and possibly a validation subset if needed. Each model will be trained on the training data and evaluated on the same test split to ensure fair comparison.

---

## 12. Risk Scoring Layer
One of the distinguishing features of the project is the addition of a risk scoring mechanism.

### Purpose of risk scoring
The purpose of this layer is to move beyond a binary answer and support security-oriented interpretation. A message predicted as phishing with 0.51 probability and a message predicted with 0.97 probability should not be treated identically. Likewise, some messages may include several explicit security indicators and therefore deserve a higher severity level.

### Proposed risk strategy
The risk score may be based on:
- the model’s phishing probability,
- the presence of suspicious words,
- the existence of URLs,
- and the presence of urgency or credential-related terms.

### Example interpretation
A sample e-mail might be classified as phishing with a moderate probability, but if it also contains a suspicious verification request and multiple urgent prompts, the final risk level may be increased to High or Critical.

### Example risk mapping
A simple version of the mapping may be:
- **Critical:** very high phishing probability and multiple suspicious indicators
- **High:** high phishing probability or several warning features
- **Medium:** moderate probability with limited indicators
- **Low:** low phishing probability and no major warning signs

This structure strengthens the project by aligning it with information security risk management thinking.

---

## 13. Explainable AI Component
A major strength of the project is its explainability layer. Security tools are more useful when they are not only accurate but also interpretable.

### Why explainability matters
If a model marks an e-mail as phishing, the user or evaluator should be able to understand the reason. This is especially important in educational settings because the goal is not only to detect attacks but also to analyze attack indicators.

### SHAP usage
The project plans to use **SHAP (SHapley Additive exPlanations)** to analyze model outputs.

### Types of explanations
The explainability section may include:

#### Global explanation
This will show which features are generally influential across the dataset. For example, terms such as “verify,” “account,” “click here,” or “urgent” may appear as strong indicators.

#### Local explanation
This will explain why a particular e-mail was classified as phishing or legitimate. Such examples are very useful for the report and the final demonstration.

### Educational and practical value
Adding explainability makes the project stronger because it transforms it from a simple machine learning exercise into an interpretable security analysis tool.

---

## 14. Demonstration Plan
The project will include a simple but effective demonstration, most likely through a notebook cell or a small script.

### Example usage
A sample e-mail will be entered into the system, and the output will display:
- predicted class,
- phishing probability,
- final risk level,
- and top suspicious indicators.

### Example output format
A demonstration may appear as follows:

**Prediction:** Phishing  
**Probability:** 0.92  
**Risk Level:** Critical  
**Important Indicators:** verify, account, urgent, click here

This is sufficient for a course presentation because it shows the practical utility of the model without requiring a full web interface.

---

## 15. Expected Deliverables
At the end of the project, the following deliverables are expected:

1. **Project report** explaining the background, methodology, experiments, results, and conclusions.
2. **Source code** including preprocessing, training, evaluation, risk scoring, and explanation steps.
3. **Experimental outputs** such as metric tables, confusion matrices, and SHAP visualizations.
4. **A simple demonstration** that shows how the system analyzes a sample e-mail.
5. **Presentation slides** summarizing the project for classroom delivery.

---

## 16. Proposed Implementation Plan
The implementation will proceed in a step-by-step manner so that the project remains manageable.

### Phase 1 – Dataset preparation
- Load the selected dataset
- Inspect class labels and text structure
- Clean missing or duplicated records
- Prepare a consistent text field

### Phase 2 – Preprocessing and feature extraction
- Normalize the text
- Apply TF-IDF vectorization
- Extract basic security-related indicators
- Prepare feature matrices for model training

### Phase 3 – Model training
- Train Naive Bayes as baseline
- Train Logistic Regression as improved model
- Train Random Forest as alternative comparison model

### Phase 4 – Evaluation
- Compare all models using core metrics
- Build confusion matrices
- Select the most suitable model for risk scoring and explainability

### Phase 5 – Risk scoring
- Convert prediction probability into meaningful risk categories
- Enhance the score using simple security-related rules if necessary

### Phase 6 – Explainability
- Apply SHAP analysis
- Produce both global and local explanation outputs

### Phase 7 – Demonstration and reporting
- Prepare example e-mails for demo
- Write the final report
- Prepare presentation materials

---

## 17. Limitations
Like any academic prototype, this project has limitations.

### Main limitations
- It focuses primarily on textual e-mail analysis.
- It does not inspect attachments or execute malware detection.
- It does not use live domain reputation services.
- It may not fully capture highly obfuscated or visually deceptive phishing attempts.
- It is not a deployed enterprise mail security solution.

These limitations are acceptable because the purpose of the project is to create a focused, interpretable, and achievable prototype within the course scope.

---

## 18. Future Work
If this project were to be extended in the future, several improvements could be explored.

### Possible future enhancements
- use of transformer-based language models,
- analysis of malicious attachments,
- integration of URL reputation and domain intelligence,
- testing against adversarially modified phishing messages,
- deployment as a real-time e-mail security filter,
- and building a simple user interface or web application around the model.

Including a future work section strengthens the academic structure of the report and shows awareness of the broader security landscape.

---

## 19. Conclusion
This project proposes a practical and academically grounded approach to phishing e-mail detection using machine learning. Its value comes not only from classification performance but also from its combination of three important aspects: detection, risk interpretation, and explainability.

By comparing multiple lightweight models, integrating risk scoring, and applying explainable AI methods, the project aims to become more than a simple text classification exercise. It becomes a compact but meaningful security decision-support prototype that is directly aligned with key topics in the computer security course.

In summary, the project is expected to demonstrate that even a relatively lightweight and manageable implementation can provide strong educational value, meaningful technical analysis, and clear relevance to real-world security problems.

