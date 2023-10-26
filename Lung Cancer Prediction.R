#https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer

# Load the necessary libraries
library(caTools)
library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)
library(caret)

# Load the dataset
pd <- read.csv("survey lung cancer.csv")

# Step 3: Handling missing values (if any)
# Check for NAs
if (sum(is.na(pd)) > 0) {
  pd <- na.omit(pd) # Remove rows with NAs if present
}
str(pd)


library(ggplot2)

# Function to create a bar plot for each categorical variable
create_bar_plot <- function(data, x_var, target_var) {
  p <- ggplot(data, aes_string(x = x_var, fill = target_var)) +
    geom_bar(position = "dodge") +
    labs(title = paste("Bar Plot of", x_var, "vs. LUNG_CANCER"),
         x = x_var, y = "Count") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p)
}

# Create individual bar plots for each categorical variable
for (col in names(pd)) {
  if (is.factor(pd[[col]]) && col != "LUNG_CANCER") {
    p <- create_bar_plot(pd, col, "LUNG_CANCER")
    print(p)
  }
}


# Step 4: Feature Engineering and Encoding Categorical Variables (if necessary)
# No specific feature engineering needed in this case, as the data seems to be in a usable format.
# If there were any categorical variables with more than 2 levels, we would have to one-hot encode them.

# Step 5: Split the data into training and testing sets
set.seed(123) # Set seed for reproducibility
split <- sample.split(pd$LUNG_CANCER, SplitRatio = 0.7) # 70% for training, 30% for testing
train_data <- subset(pd, split == TRUE)
test_data <- subset(pd, split == FALSE)

# Step 5 (continued): Convert "YES" and "NO" to binary (0 and 1) for logistic regression
train_data$LUNG_CANCER <- ifelse(train_data$LUNG_CANCER == "YES", 1, 0)
test_data$LUNG_CANCER <- ifelse(test_data$LUNG_CANCER == "YES", 1, 0)

# Step 6 (continued): Train and Evaluate Models for Lung Cancer Prediction
# Model 1: Logistic Regression
model_logistic <- glm(LUNG_CANCER ~ ., data = train_data, family = binomial)
logistic_predictions <- predict(model_logistic, newdata = test_data, type = "response")

# Convert the predicted probabilities to binary (0 and 1) based on a threshold (e.g., 0.5)
logistic_predictions <- ifelse(logistic_predictions >= 0.5, 1, 0)

# Convert both predicted and actual values to factors with the same levels (0 and 1)
logistic_predictions <- factor(logistic_predictions, levels = c(0, 1))
test_data$LUNG_CANCER <- factor(test_data$LUNG_CANCER, levels = c(0, 1))

# Model 2: Decision Trees
model_decision_tree <- rpart(LUNG_CANCER ~ ., data = train_data, method = "class")
decision_tree_predictions <- predict(model_decision_tree, newdata = test_data, type = "class")


# Convert target variable back to factor format
train_data$LUNG_CANCER <- factor(train_data$LUNG_CANCER, levels = c(0, 1))
test_data$LUNG_CANCER <- factor(test_data$LUNG_CANCER, levels = c(0, 1))

# Model 3: Random Forest
model_random_forest <- randomForest(LUNG_CANCER ~ ., data = train_data, ntree = 100)
random_forest_predictions <- predict(model_random_forest, newdata = test_data)


# Set factor levels explicitly for both predicted and actual values
levels_rf <- levels(random_forest_predictions)
levels_actual <- levels(test_data$LUNG_CANCER)
random_forest_predictions <- factor(random_forest_predictions, levels = levels_actual)

# Model 4: Support Vector Machines (SVM)
model_svm <- svm(LUNG_CANCER ~ ., data = train_data)
svm_predictions <- predict(model_svm, newdata = test_data)

# Set factor levels explicitly for both predicted and actual values
svm_predictions <- factor(svm_predictions, levels = levels_actual)

# Step 7: Evaluation of Models
# We can use various evaluation metrics like accuracy, confusion matrix, precision, recall, F1-score, etc.
library(caret)

# Function to calculate and print evaluation metrics
calculate_metrics <- function(predictions, actual) {
  confusion <- confusionMatrix(predictions, actual)
  accuracy <- confusion$overall["Accuracy"]
  precision <- confusion$byClass["Pos Pred Value"]
  recall <- confusion$byClass["Sensitivity"]
  f1_score <- confusion$byClass["F1"]
  
  cat("Accuracy:", round(accuracy, 2), "\n")
  cat("Precision:", round(precision, 2), "\n")
  cat("Recall:", round(recall, 2), "\n")
  cat("F1-Score:", round(f1_score, 2), "\n")
}

# Evaluate Logistic Regression Model
cat("Logistic Regression Model:\n")
calculate_metrics(logistic_predictions, test_data$LUNG_CANCER)

# Evaluate Decision Tree Model
cat("Decision Tree Model:\n")
calculate_metrics(decision_tree_predictions, test_data$LUNG_CANCER)
# Plot the Decision Tree
rpart.plot(model_decision_tree, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)

# Evaluate Random Forest Model
cat("Random Forest Model:\n")
calculate_metrics(random_forest_predictions, test_data$LUNG_CANCER)

# Evaluate SVM Model
cat("Support Vector Machines (SVM) Model:\n")
calculate_metrics(svm_predictions, test_data$LUNG_CANCER)


  
 