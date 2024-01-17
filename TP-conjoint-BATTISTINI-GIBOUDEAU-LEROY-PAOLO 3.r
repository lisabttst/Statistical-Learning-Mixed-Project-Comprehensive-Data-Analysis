rm(list=ls())
install.packages('party')
install.packages('caret')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('ggplot2')
install.packages('rattle')
install.packages('e1071')
install.packages("Metrics")
install.packages("isotree")
install.packages("plotrix")
library(party)
library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(rattle)
library(e1071)
library(Metrics)
library(pROC)
library(Rarity)
library(isotree)
library(MASS)

source("ACP_param.R")
source("quality_ACP.R")
source("AFD.R")

# Question 1 : analyse statistique

Data <- read.csv('Income_Inequality.csv',sep = ';')
str(Data)
parametres <- names(Data)[1:22]
print(parametres)
missing_values <- sapply(Data, function(x) sum(is.na(x)))
print(missing_values) 

summary(Data)
table(Data$Income_Inequality)

max_val <- apply(Data[, 4:22], 2, max)
ordegrandeur<- floor(log10(max_val))
print(ordegrandeur)
par(mfrow=c(1,6))
boxplot(Data[, 4:6], main="Boxplots of Columns 4, 5 and 6", las=2)
boxplot(Data[, c(7,18:22)], main="Boxplots of Column 7, 15:19", las=2)
boxplot(Data[, 8:10], main="Boxplots of Column 8,9, 10", las=2)
boxplot(Data[, c(11,14,15,16,17)], main="Boxplots of Columns 11, 14:17", las=2)
boxplot(Data[, 12], main="Boxplots of Column 12 = finance1", las=2)
boxplot(Data[, 13], main="Boxplots of Column 13 = finance2", las=2)
par(mfrow=c(1,1))

cor_matrice <- cor(Data[, sapply(Data, is.numeric)]) 
print(cor_matrice)
corPlot(cor_matrice, method = 'pearson')

skewness_vals <- sapply(Data[, sapply(Data, is.numeric)], skewness)
kurtosis_vals <- sapply(Data[, sapply(Data, is.numeric)], kurtosis)
print(skewness_vals[3:21])
print(kurtosis_vals[3:21])


#question 2 a

set.seed(1234) 
Data <- Data[, -c(1, 2)]
splitIndex <- createDataPartition(Data$Income_Inequality, p = 0.7, list = FALSE)
trainData <- Data[splitIndex, ]
testData <- Data[-splitIndex, ]


# questiom 2 b 

#Construction et optimisation de l'arbre en utilisant la validation croisée
fitControl <- trainControl(method = "cv", number = 5)
fit <- train(Income_Inequality ~ ., data=trainData, method="rpart", trControl=fitControl,
             tuneGrid=data.frame(cp=seq(0.01, 0.5, 0.01)))
best_cp <- fit$bestTune$cp
tree <- rpart(Income_Inequality ~ ., data=trainData, cp=best_cp)
 
# Visualiser l'arbre élagué
windows(rescale="fit", width=6, height=6)
rpart.plot(tree)

#predictions
predictions <- predict(tree, testData, type="class")

#Question 2c
predictions <- factor(predictions)
#Matrice de confusion
testData$Income_Inequality <- factor(testData$Income_Inequality)
cm <- confusionMatrix(predictions, testData$Income_Inequality)
print(cm)
#Detail by class
cm$byClass
#CourbeROC
prob_predictions <- predict(tree, testData, type="prob")
roc_obj <- roc(response=testData$Income_Inequality, predictor=prob_predictions[, "H"])
# L'AUC
auc(roc_obj)
# Plot ROC
plot(roc_obj, main="ROC Curve", col="#1c61b6", lwd=2)
abline(a=0, b=1, lwd=2, lty=2, col="red")

#QUESTION 3
# Construire la forêt d'isolement
iso_forest <- isolation.forest(Data)

# Calculer les scores d'anomalie pour chaque observation
anomaly_scores <- predict(iso_forest, Data)

# Trier les scores d'anomalie et obtenir les indices des 10 scores les plus élevés et les plus bas
top_10_anomalies_indices <- order(anomaly_scores, decreasing = TRUE)[1:10]
bottom_10_anomalies_indices <- order(anomaly_scores)[1:10]
# Extraire les observations correspondantes
top_10_anomalies <- Data[top_10_anomalies_indices, ]
bottom_10_anomalies <- Data[bottom_10_anomalies_indices, ]

# Afficher les observations
print("Top 10 Anomalies:")
print(top_10_anomalies)
print("Bottom 10 Anomalies:")
print(bottom_10_anomalies)

# Supprimer les 50 observations ayant les scores d'anomalie les plus élevés
indices_to_remove <- order(anomaly_scores, decreasing = TRUE)[1:50]
Data_cleaned <- Data[-indices_to_remove, ]

# Diviser les données nettoyées en ensembles d'entraînement et de test
set.seed(1234) 
splitIndex_cleaned <- createDataPartition(Data_cleaned$Income_Inequality, p = 0.7, list = FALSE)
trainData_cleaned <- Data_cleaned[splitIndex_cleaned, ]
testData_cleaned <- Data_cleaned[-splitIndex_cleaned, ]

# Construire et visualiser un nouvel arbre de décision avec les données nettoyées
fitControl_cleaned <- trainControl(method = "cv", number = 5)
fit_cleaned <- train(Income_Inequality ~ ., data = trainData_cleaned, method = "rpart", trControl = fitControl_cleaned, tuneGrid = data.frame(cp = seq(0.01, 0.5, 0.01)))
best_cp_cleaned <- fit_cleaned$bestTune$cp
tree_cleaned <- rpart(Income_Inequality ~ ., data = trainData_cleaned, cp = best_cp_cleaned)
dev.off()
rpart.plot(tree_cleaned, main="Arbre de Décision Nettoyé")

# Prédictions avec l'arbre nettoyé
predictions_cleaned <- predict(tree_cleaned, testData_cleaned, type = "class")

#Matrice de confusion
# Convertir les prédictions et les vraies valeurs en facteurs
predictions_cleaned <- factor(predictions_cleaned)
testData_cleaned$Income_Inequality <- factor(testData_cleaned$Income_Inequality)

# Calculer la matrice de confusion
predictions_cleaned <- factor(predictions_cleaned)
testData_cleaned$Income_Inequality <- factor(testData_cleaned$Income_Inequality)
cm_cleaned <- confusionMatrix(predictions_cleaned, testData_cleaned$Income_Inequality)
print(cm_cleaned)

#Detailed Accuracy By Class
cm_cleaned$byClass

#Courbe ROC et AUC
# Prédictions de probabilité
prob_predictions_cleaned <- predict(tree_cleaned, testData_cleaned, type="prob")

# Calcul de la courbe ROC et de l'AUC
roc_obj_cleaned <- roc(response = testData_cleaned$Income_Inequality, predictor = prob_predictions_cleaned[, "H"])
auc_cleaned <- auc(roc_obj_cleaned)
auc_cleaned

# Tracer la courbe ROC
plot(roc_obj_cleaned, main="Courbe ROC - Arbre Nettoyé", col="#1c61b6", lwd=2)
abline(a=0, b=1, lwd=2, lty=2, col="red")


# Question 4 : ACP

# Récupération des données 
trainDataACP <- trainData[,2:ncol(trainData)]

# Test d'ACP pour déterminer le nombre de composantes à retenir
acp <- ACP(trainDataACP, norm = TRUE)
q_2 <- quality_ACP(acp$Val_p, acp$Comp, acp$Vect_p, norm=TRUE)
q_3 <- quality_ACP(acp$Val_p, acp$Comp, acp$Vect_p, norm=TRUE)
q_4 <- quality_ACP(acp$Val_p, acp$Comp, acp$Vect_p, norm=TRUE)
q_5 <- quality_ACP(acp$Val_p, acp$Comp, acp$Vect_p, norm=TRUE)
q_6 <- quality_ACP(acp$Val_p, acp$Comp, acp$Vect_p, norm=TRUE)
dev.off()
par(mfrow=c(2,3))
boxplot(q_2$Qual, main="k=2")
boxplot(q_3$Qual, main="k=3")
boxplot(q_4$Qual, main="k=4")
boxplot(q_5$Qual, main="k=5")
boxplot(q_6$Qual, main="k=6")
par(mfrow=c(2,3))

H <- acp$Comp[trainData[,1]=="H",]
L <- acp$Comp[trainData[,1]=="L",]
plot(H[,1], H[,2], col="red", xlab="CS1", ylab="CS2")
points(L[,1], L[,2], col="blue")
plot(H[,2], H[,3], col="red", xlab="CS2", ylab="CS3")
points(L[,2], L[,3], col="blue")
plot(H[,3], H[,4], col="red", xlab="CS3", ylab="CS4")
points(L[,4], L[,4], col="blue")
plot(H[,1], H[,4], col="red", xlab="CS1", ylab="CS4")
points(L[,1], L[,4], col="blue")
plot(H[,2], H[,4], col="red", xlab="CS2", ylab="CS4")
points(L[,2], L[,4], col="blue")
legend("topleft", legend = c("H", "L"), fill = c("red", "blue"), xpd= TRUE, inset=c(-0.1, -0.2))


# Projette les 30% tests sur les 2 premiers axes factoriels
dev.off()
testDataACP <- as.matrix(testData[,2:ncol(trainData)])
for (i in 1:ncol(testDataACP)) {
  testDataACP[,i] <- (testDataACP[,i] - mean(testDataACP[,i])) / sd(testDataACP[,i])
}

comptest <- testDataACP %*% acp$Vect_p
Htest <- comptest[testData[,1]=="H",]
Ltest <- comptest[testData[,1]=="L",]
plot(Htest[,1], Htest[,2], col="red")
points(Ltest[,1], Ltest[,2], col="blue")


qtest <- quality_ACP(acp$Val_p, comptest, acp$Vect_p, norm=TRUE)
par(mfrow=c(1,1))
barplot(qtest$Qual, main="Qualité de projection des individus")
pires_ind <- order(qtest$Qual)[1:10]
print(pires_ind)
print(Data[pires_ind,])


# Question 5 : AFD

# Fournit les matrices W, B et V
val <- trainData[,1]
trainDataAFD <- trainData[,2:ncol(trainData)]
trainDataAFD <- cbind(trainDataAFD, val)

afd <- AFD(trainDataAFD, c("H", "L"), 20, diag_V=TRUE, norm=TRUE)
B <- afd$B
V <- afd$V
W <- afd$W


# Détermine les composantes
comp <- as.matrix(afd$data[,-ncol(afd$data)]) %*% afd$v
H <- comp[trainData[,1]=="H",]
L <- comp[trainData[,1]=="L",]

stripchart(H[,1], col="red", xlim=c(-4,4), main="Graphe 1D")
stripchart(L[,1], col="blue", add=TRUE)

plot(H[,1], H[,2], col="red", xlab="ax1", ylab="ax2", main="Premier plan factoriel")
points(L[,1], L[,2], col="blue")


# AFD sur les données de test
val_test <- testData[,1]
afd_test_result <- MASS::lda( testData[,2:ncol(testData)], val_test)
print(afd_test_result)

scores <- predict(afd_test_result)$x

mahalanobis_distances <- mahalanobis(scores, colMeans(scores), cov(scores))
print("Distances de Mahalanobis :")
barplot(mahalanobis_distances, main="Distance de Mahalanobis pour les individus")
order(mahalanobis_distances)[1:10]


# Question 6 : A. fonction discriminante prédictive
afd_predict <- MASS::lda(trainData[,2:ncol(trainData)], val)
predictions_afd <- predict(afd_predict, testData[,2:ncol(testData)])
print(predictions_afd$class)

H1 <- predictions_afd$x[predictions_afd$class=="H",]
L1 <- predictions_afd$x[predictions_afd$class=="L",]

stripchart(H1, col="red", xlim=c(-4,4), main="Prédictions avec le modèle entrainé")
stripchart(L1, col="blue", add=TRUE)

print(predictions_afd$class)
plot(predictions_afd$posterior[,1], col="red", main="Probabilités à posteriori")
points(predictions_afd$posterior[,2], col="blue")


real_classes <- testData$Income_Inequality
goods <- 0
for (i in 1:length(predictions_afd$class)) {
  if (predictions_afd$class[i] == real_classes[i]) {
    goods <- goods + 1
  }
}
cat("Accuracy :", goods/length(real_classes))

cm_afd <- confusionMatrix(predictions_afd$class, testData$Income_Inequality)
print("Matrice de confusion - AFD:")
print(cm_afd)

#Question 6B : arbre de décision avec les données de l'ACP
# Utilisation directe des scores des composantes principales
trainData_reduced <- acp$Comp[, 1:4]

# Ajout de la variable cible à ces données réduites
trainData_reduced <- data.frame(trainData_reduced)
trainData_reduced$Income_Inequality <- trainData$Income_Inequality


# Construction de l'arbre de décision
fitControl <- trainControl(method = "cv", number = 5)
fit <- train(Income_Inequality ~ ., data = trainData_reduced, method = "rpart", trControl = fitControl, tuneGrid = data.frame(cp = seq(0.01, 0.5, 0.01)))
best_cp <- fit$bestTune$cp
tree <- rpart(Income_Inequality ~ ., data = trainData_reduced, cp = best_cp)

# Visualisation de l'arbre
rpart.plot(tree)

# Prédictions sur l'ensemble de test réduit
# Utilisation directe des scores des composantes principales
testData_reduced <- comptest[,1:4]
  
# Ajout de la variable cible à ces données réduites
testData_reduced <- data.frame(testData_reduced)
predictions <- predict(tree, testData_reduced, type="class")

# Conversion des prédictions en facteurs
predictions <- factor(predictions, levels=levels(testData$Income_Inequality))

# Matrice de confusion pour évaluer la performance

cm <- confusionMatrix(predictions, testData$Income_Inequality)
print(cm)



#Question 6B : AFD prédictive
# Assurez-vous que la variable cible est incluse
trainData_reduced$Income_Inequality <- trainData$Income_Inequality
testData_reduced$Income_Inequality <- testData$Income_Inequality
# Construction du modèle AFD
afd_model <- MASS::lda(Income_Inequality ~ ., data = trainData_reduced)
# Prédictions AFD sur l'ensemble de test réduit
predictions_afd_reduced <- predict(afd_model, testData_reduced[, -ncol(testData_reduced)])$class
predictions_afd_reduced <- factor(predictions_afd_reduced, levels=levels(testData_reduced$Income_Inequality))

# Matrice de confusion pour l'AFD
cm_afd_reduced <- confusionMatrix(predictions_afd_reduced, testData_reduced$Income_Inequality)
print("Matrice de confusion - AFD sur données réduites:")
print(cm_afd_reduced)

