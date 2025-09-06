# 🔐 Academic Stress Detector - Model Service

## 💡 Overview

Model Service of this application is splitted in two parts, first the api which was made in FastAPI. Having in mind that api was mode in Python we needed to create model also in Python specificly in pytorch and pytorch lightning. Model is basic neural network with sigmoid activation functions and couple of dropout and linear layers, which allows us to predict student stress level based on it's current data.

![Banner](./assets/application-tree.png)

## 🪾 Branches
* `/api/model/predict` prediction branch

    ```
    body: {
        headache: int
        sleepQuality: int
        breathingProblems: int
        noiseLevel: int
        livingConditions: int
        safety: int
        basicNeeds: int
        academicPerformance: int
        studyLoad: int
        teacherStudentRelationship: int
        futureCareerConcerns: int
        socialSupport: int
        peerPressure: int
        extracurricularActivities: int
        bullying: int
    }
    ```

## 🎯 Model Training

Model was trained, with such hyperparameters:
```python
learning_rate = 1e-4
dropout = 0.5
gamma = 0.995
num_epochs = 100
hidden_size = 128
num_classes = 3
```

## 🗒️ Features
* Sigmoid activation function;
* Cross entropy loss;
* Adam optimizer;
* Exponential learning rate scheduler;
* Training progress bar;
* Tensorboard integration;
* pydantic scheme validation;

## ⚙️ Command Tools

To work with this project locally or in a containerized environment, use the following commands:
```bash
./mvnw spring-boot:run # to run project

./mvnw clean # clean logs


export {PROPERTY_NAME}={VALUE} # set env variable 
````

## 🧠 Tech Stack
<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,fastapi,pytorch,sklearn,git" />
  </a>
</p>