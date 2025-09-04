from pydantic import BaseModel

class StressInput(BaseModel):
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