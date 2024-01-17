data = read.csv("Ionosphere.csv",sep=";")

library("solitude")
library("tidyverse")
library("mlbench")

data=data[,c(1:32)]
data <- na.omit(data)

splitter   = data %>%
  rsample::initial_split(prop = 0.7)
pima_train = rsample::training(splitter)
pima_test  = rsample::testing(splitter)

iso = isolationForest$new(sample_size = 245)
iso$fit(pima_train)

scores_train = pima_train %>%
  iso$predict() %>%
  arrange(desc(anomaly_score))

scores_train

scores_test = pima_test %>%
  iso$predict() %>%
  arrange(desc(anomaly_score))

scores_test
