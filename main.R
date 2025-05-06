# https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data

# Problem Statement: year is 2010
#  HELP International have been able to raise around $ 10 million. 
# Now the CEO of the NGO needs to decide how to use this money strategically 
# and effectively. So, CEO has to make decision to choose the countries that are 
# in the direst need of aid. Hence, your Job as a Data scientist is to categorize
# the countries using some socio-economic and health factors that determine the 
# overall development of the country. Then you need to suggest the countries 
# which the CEO needs to focus on the most. '''


library(corrplot)
library(car)
library(Hmisc) # for redundancy analysis
library(ggplot2)
library(gridExtra)
library(dplyr)
library(plotly)
library(glmnet) # for ridge regression
library(reshape2) # for reshaping the data easily
library(viridis) # for color palettes
library(tidyr)
library(ggrepel)
library(patchwork)
library(RColorBrewer)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(countrycode)
library(scales)
library(glmnet)
library(broom) # for tidy() function
library(tibble) # for rownames_to_column()
library(rpart)         # recursive partitioning for regression trees
library(rpart.plot)    # for nicer tree plots (optional)
library(randomForest) # for random forests


data <- read.csv("data/Country-data.csv", sep = ",", header = TRUE, na.strings = c("", "NA", "NULL")) # Read the CSV file
dict <- read.csv("data/data-dictionary.csv", sep = ",", header = TRUE, na.strings = c("", "NA", "NULL")) # Read the CSV file

summary(data)
str(data)
sapply(data, function(x) sum(is.na(x))) # Check for missing values applying the function sum() to each column of data
sapply(data, class) # Check for data types


# Histograms, boxplots: check variables' distributions

par(mfrow=c(3, 3), mar=c(2, 2, 2, 2), oma=c(0, 0, 3, 0)) # 3 rows, 3 columns
for(i in 2:10) {
  hist(data[, i], 
       main=names(data)[i], 
       col="lightblue")}
mtext("Variables Original Distribution", outer=TRUE, cex=1.0, font=2)
for(i in 2:10) {
  boxplot(data[,i], 
          main = names(data)[i], 
          col = "lightblue")}
mtext("Variables Box Plot", outer=TRUE, cex=1.0, font=2)


# Understand variables' normality

# Set up a 3x3 grid layout for plotting (3 rows and 3 columns per page)
# This means up to 9 plots will be displayed at once
par(mfrow = c(3, 3))

# Loop through each column name in the dataset 'data'
# lapply() applies a function to each element of names(data) (i.e., each variable name)
lapply(names(data[2:10]), function(var) { 
  
  # Create a Q-Q (Quantile-Quantile) plot for the current variable:
  # - Compares sample quantiles to theoretical quantiles of a normal distribution
  # - Helps check if the variable follows a normal distribution
  qqnorm(data[[var]],  # Plot data from the current variable
         main = paste("Q-Q Plot of", var),  # Dynamic title      # No main title (empty string)
         xlab = '')      # No x-axis label (empty string)
  
  # Add a reference line to the Q-Q plot:
  # - A straight line is expected if data is normally distributed
  # - Points deviating from the line suggest non-normality
  qqline(data[[var]],  # Draw reference line using current variable's data
         col = 'lightblue') # Make the line light blue for visibility
  
  # Return NULL invisibly (suppresses lapply() from printing output)
  # We only care about the plots, not the return values
  invisible(NULL)
})


# Perform Shapiro-Wilk test for each continuous variable
set.seed(123)
shapiro_results <- lapply(names(data[2:10]), function(var) {
  test_result <- shapiro.test(data[[var]])
  c(Statistic = test_result$statistic, P_Value = test_result$p.value)
})

shapiro_df <- as.data.frame(do.call(rbind, shapiro_results))
colnames(shapiro_df) <- c("Statistic (W)", "P_Value")
rownames(shapiro_df) <- names(data[2:10])
shapiro_df$TestResult <- ifelse(shapiro_df$P_Value <= 0.01, "Reject", "Fail to Reject")
shapiro_df
# Perform Shapiro-Wilk test for each continuous variable - Log transformed


# Apply log(x + 1) transformation to each element (column) in the 'data' list or data frame
# Since countries like Czech Republic, Ireland, Japan and Seychelles have negative inflation, and the lowest is -4,
# we shift the values by adding 5 to avoid taking log of negative numbers
data_log <- lapply(data[2:10], function(x) log(x + 5))

# For each variable (column) name in the transformed data, perform a Shapiro-Wilk normality test
shapiro_results_log <- lapply(names(data_log), function(var) {
  test_result <- shapiro.test(data_log[[var]])  # Run the test on the column
  c(Statistic = test_result$statistic, P_Value = test_result$p.value)  # Extract statistic and p-value
})

# Convert the list of results into a data frame, row-binding each result into a single table
shapiro_df_log <- as.data.frame(do.call(rbind, shapiro_results_log))

# Rename the columns for clarity
colnames(shapiro_df_log) <- c("Statistic (W)", "P_Value")

# Set the row names of the data frame to be the variable names
rownames(shapiro_df_log) <- names(data_log)

# Add a new column indicating whether to reject the null hypothesis (normality) based on p-value
shapiro_df_log$TestResult <- ifelse(shapiro_df_log$P_Value <= 0.01, "Reject", "Fail to Reject")

# Output the final data frame with Shapiro-Wilk test results
shapiro_df_log


# so the only normal variable seems to be the logarithm of health spending per capita as a percentage of gdp.
# Even tho the logarithms of the variables turned out not to be normal, we can still plot the results
# to see if they look at least more normal than before:
data_log <- as.data.frame(data_log) # Convert the list of log-transformed data back to a data frame

par(mfrow=c(3, 3), mar=c(2, 2, 2, 2), oma=c(0, 0, 3, 0)) # 3 rows, 3 columns
for(i in 1:9) {
  hist(data_log[,i], 
          main = names(data)[i], 
          col = "lightblue")}
mtext("Log Variables Histograms", outer=TRUE, cex=1.0, font=2)


par(mfrow=c(3, 3), mar=c(2, 2, 2, 2), oma=c(0, 0, 3, 0)) # 3 rows, 3 columns
for(i in 1:9) {
  boxplot(data_log[,i], 
       main = names(data)[i], 
       col = "lightblue")}
mtext("Log Variables Box Plot", outer=TRUE, cex=1.0, font=2)


par(mfrow = c(1,1))# Close the current graphics device (if any) to reset plotting
# Correlation matrix
correlational_matrix <-cor(data[,2:10], use = "pairwise.complete.obs")
corrplot(correlational_matrix, 
         tl.col = "black")

# WITH THIS PLOT WE CAN SEE HOW CHILD MORTALITY IS STRONGLY CORRELATED WITH OTHER VARIABLES


cor_matrix <- cor(data_log[,1:9], use = "pairwise.complete.obs")  # Check correlation matrix
corrplot(cor_matrix, 
         tl.col = "black", main = "Log Correlation Matrix")  # Plot correlation matrix

# Covariance Matrix
cov_matrix <- cov(data[2:10])
round(cov_matrix, 4)
round(cor_matrix, 4)



# VIF implementation

predictors <- data[-1]  # Excludes first column

# Calculate VIF - method 1 (using first remaining column as placeholder)
vif_model <- lm(
  as.formula(paste(names(predictors)[1], "~ .")),  # First predictor as placeholder
  data = predictors
)
vif_values <- car::vif(vif_model)


# Create results table
vif_df <- data.frame(
  variable = names(vif_values),
  vif = vif_values,
  vif_status = ifelse(vif_values > 5, ifelse(vif_values > 10, "High VIF (Issue)", "Concernably high"), "OK"),
  row.names = NULL)

# View results
print(vif_df)


### COMPUTE VIF ON THE LOGS
predictors_log <- data_log[-1]  # Excludes first column

# Calculate VIF - method 1 (using first remaining column as placeholder)
vif_model <- lm(
  as.formula(paste(names(predictors_log)[1], "~ .")),  # First predictor as placeholder
  data = predictors_log
)
vif_values <- car::vif(vif_model)

# Create results table
vif_df <- data.frame(
  variable = names(vif_values),
  vif = vif_values,
  vif_status = ifelse(vif_values > 5, ifelse(vif_values > 10, "High VIF (Issue)", "Concernably high"), "OK"),
  row.names = NULL)

# View results
print(vif_df)


#relationships were nonlinear. After log transformation, "income" and "child mortality" 
#became linearly predictable from other variables. This shows that the log transformation 
#revealed hidden dependencies that were not detectable before.




# Redundancy analysis on original variables did not find redundant features 
# redundancy analysis on original data
redun_result <- redun(~ ., data = data[2:10], nk = 0) 
print(redun_result)


# redundancy analysis on log
redun_result <- redun(~ ., data = data_log, nk = 0) 
print(redun_result)
# Income and child mortality are redundant with other variables


#A GENERAL INSIGHT IS THAT "income", "child mortality" and "gdpp" WERE NONLINEARLY CORRELATED WITH OTHER VARIABLES 
# WE KNOW THIS CAUSE THE RELATION WASN'T SPOTTED ON NORMAL VARIABLES BY THE VIF, BUT EASILY SPOTTED WITH THE CORRELATION 
# MATRIX AND THE VIF ON LOGARITHMIC VARIABLES, CAUSE THESE RELATIONS ARE NOT LINEAR AND THE LOG LINEARIZES THEM



# LET'S TRY TO GET A LIL BIT DEEPER INTO THE CORRELATIONS AMONG INCOME, GDPP AND CHILD MORT
# WE SPOTTED THOSE PROBLEMATIC VARIABLES THROUGH THE CORRELATION MATRIX 


# PLEASE NOTE: FOR CONSIDERATIONS ON THE DIFFERENCE BETWEEN INCOME AND GDP PER CAPITA, REFER TO
# notion page sl/UNSUP/INCOME vs GDPP


# standardize data

data_scaled <- as.data.frame(scale(data[2:10]))
data_scaled_log <- as.data.frame(scale(data_log))
 

# let's run redundancy analysis on the scaled log data

redun_result_scaled <- redun(~ ., data = data_scaled_log, nk = 0)
print(redun_result_scaled)

# we got the same results as before 

# Simple scatter plot
ggplot(data_scaled_log, aes(x = gdpp, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  labs(title = "Income vs GDP per capita", x = "Log GDP per capita standardized", y = "Log Income standardized") +
  theme_minimal()

ggplot(data_scaled_log, aes(x = gdpp, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_smooth(method = "lm", color = 'red') +
  labs(title = "Log-Log Relationship", x = "log(GDP per capita) standardized", y = "log(Income) standardized") +
  theme_minimal()
# as we can see this log-log relation is very close to being perfectly linear:
# so the original (unlogged) relationship between income and gdpp is a power law
# income ∝( gdpp)^β where β  is the slope of of the log-log line
# Now report the slope of the regression line (Beta)
summary(lm(income ~ gdpp, data = data_scaled_log))

# the coefficient is highly significantly equal to 9.720e-01 and the 
# intercept is 9.044e-16, almost 0 and not significant

# If log(Income) = α + β * log(GDPP), then

# Income = exp(α) × GDPP^β

# Income≈(GDP per capita) ^ 0.972


# Correct intercept and beta from your model
# Define alpha and beta FIRST
alpha <- 9.044e-16  # your intercept (very small)
beta <- 0.972       # your slope

# Plot on original data
ggplot(data, aes(x = gdpp, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  stat_function(fun = function(x) exp(alpha) * x^beta, color = 'red', size = 1.2) +
  labs(
    title = "Income vs GDP per capita (Original Scale with Correct Power-Law Fit)",
    x = "GDP per capita",
    y = "Income"
  ) +
  theme_minimal()

# not satisfied with the power law fit, let's try to fit a polynomial regression on original data
# Fit quadratic model
poly2_model <- lm(income ~ poly(gdpp, 2, raw = TRUE), data = data)

# Plot
ggplot(data, aes(x = gdpp, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2, raw = TRUE), color = 'red', size = 1.2) +
  labs(
    title = "Quadratic Fit: Income vs GDP per capita",
    x = "GDP per capita",
    y = "Income"
  ) +
  theme_minimal()

# Fit quadratic model on data_scaled_log
poly2_model_log <- lm(income ~ poly(gdpp, 2, raw = TRUE), data = data_scaled_log)

# Plot
ggplot(data_scaled_log, aes(x = gdpp, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2, raw = TRUE), color = 'red', size = 1.2) +
  labs(
    title = "Quadratic Fit: Income vs GDP per capita",
    x = "GDP per capita",
    y = "Income"
  ) +
  theme_minimal()

# We can clearly see a correlation spotted by correlation matrices.


# now to officially say that these two are not linearly correlated, I will run a linear model
# and compare the adjusted R^2:
linear_model <- lm(income ~ gdpp, data = data_log)
summary(linear_model)
summary(poly2_model)
summary(poly2_model_log) # adj R^2 = 0.951 which brought to a high vif(?)

# matter of facts, the adj R^2 of the polynomial model is 0.8194 and is better than the linear 
# one of 0.8008 and the RSE is lower on the polynomial (8193 compared to 8603)


# same with child_mort


# Simple scatter plot
ggplot(data_scaled_log, aes(x = child_mort, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  labs(title = "Income vs Child Mortality", x = "Child Mortality", y = "Income") +
  theme_minimal()

ggplot(data_scaled_log, aes(x = child_mort, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_smooth(method = "lm", color = 'red') +
  labs(title = "Log-Log Relationship", x = "log(Child Mortality) standardized", y = "log(Income) standardized") +
  theme_minimal()

# Report the slope of the regression line
summary(lm(income ~ child_mort, data = data_scaled_log))

# Define alpha and beta
alpha <- coef(lm(income ~ child_mort, data = data_scaled_log))[1]
beta <- coef(lm(income ~ child_mort, data = data_scaled_log))[2]

# Plot fitted power-law on original data
ggplot(data, aes(x = child_mort, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  stat_function(fun = function(x) exp(alpha) * x^beta, color = 'red', size = 1.2) +
  labs(
    title = "Income vs Child Mortality (Original Scale with Power-Law Fit)",
    x = "Child Mortality",
    y = "Income"
  ) +
  theme_minimal()

# Fit quadratic model
poly2_model <- lm(income ~ poly(child_mort, 2, raw = TRUE), data = data)

# Plot quadratic fit
ggplot(data, aes(x = child_mort, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2, raw = TRUE), color = 'red', size = 1.2) +
  labs(
    title = "Quadratic Fit: Income vs Child Mortality",
    x = "Child Mortality",
    y = "Income"
  ) +
  theme_minimal()
# Fit quadratic model on logs 
poly2_model <- lm(income ~ poly(child_mort, 2, raw = TRUE), data = data_scaled_log)

# Plot quadratic fit
ggplot(data_scaled_log, aes(x = child_mort, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2, raw = TRUE), color = 'red', size = 1.2) +
  labs(
    title = "Quadratic Fit: Income vs Child Mortality",
    x = "Child Mortality",
    y = "Income"
  ) +
  theme_minimal()

# Compare linear vs quadratic model
linear_model <- lm(income ~ child_mort, data = data)
summary(linear_model)
summary(poly2_model)
summary(poly2_model_log) 
# also here the adj r^2 of the polynomial model is better than the linear one 
# 0.380 vs 0.270



# Let's do the PCA on the scaled log data now:

pca_result <- prcomp(data_scaled_log)
summary(pca_result)


# PCA from old project

pca_result_var <- pca_result$sdev^2 # variance of each principal component
pca_var_per <- round(pca_result_var/sum(pca_result_var)*100, 2) # percentage of variance explained by each component

par(mfrow = c(1,2), mar = c(5,5,4,2)) # mar = bottom, left, top, right
scree_plot <-barplot(pca_var_per, 
                     main = "Scree Plot", 
                     xlab = "Principal Components",
                     ylab = "Percent Variation", 
                     names.arg = paste("PC", 1:length(pca_var_per)),
                     col = "lightblue")


# Add a line to indicate the cumulative variance
cumvar <- cumsum(pca_var_per) # cumulative variance per each component
residvar <- 100 - cumvar # residual variance

elbow_plot <- plot(1:length(residvar), residvar, 
                   type = "n", 
                   main = "Elbow Plot - Residual Variance Plot", 
                   xlab = "Number of Principal Components",
                   ylab = "Residual Variance (%)", 
                   col = "lightblue")
lines(1:length(residvar), residvar, col = 'black')  
points(1:length(residvar), residvar, col = 'lightblue', pch = 19) 

optimal_pcs_num <- 4
points(optimal_pcs_num, residvar[optimal_pcs_num], col = '#D22B2B', pch = 19, cex = 2) # highlight the optimal number of components
text(optimal_pcs_num, residvar[optimal_pcs_num], , labels = paste(optimal_pcs_num, 'PCs ~ 90.81% Variance Explained'), pos = 3, col = '#D22B2B')


# Set plotting layout to 1x1 (one plot per window)
par(mfrow = c(1,1)) 

# Rename rows of PCA scores with simple numbers
rownames(pca_result$x) <- 1:nrow(pca_result$x) 

# Convert PCA scores to a data frame
pca_data <- as.data.frame(pca_result$x)

# Create a data frame summarizing sample coordinates on the first 4 principal components
pca.data.summary <- data.frame(  # creates a dataframe where each row is a sample from my data
  sample = rownames(pca_result$x), # and as columns there are its coordinates in Principal Component space
  X = pca_result$x[,1],
  Y = pca_result$x[,2],
  Z = pca_result$x[,3],
  W = pca_result$x[,4],
  row.names = NULL # row names are set to NULL to avoid confusion
)

# Create a data frame summarizing feature loadings on the first 4 principal components
pca.loadings <- data.frame(
  variable = names(data_log), # loadings tell you how much each original feature
  X = pca_result$rotation[,1],   # contributed to each PC. 
  Y = pca_result$rotation[,2],   # Technically loadings are the correlation from each original 
  Z = pca_result$rotation[,3],   # variable and the principal components
  W = pca_result$rotation[,4],
  row.names = NULL
)


# Normalize sample scores to fit within [-1,1] for better visualization
pca.data.summary$X <- pca.data.summary$X / max(abs(pca.data.summary$X))
pca.data.summary$Y <- pca.data.summary$Y / max(abs(pca.data.summary$Y))
pca.data.summary$Z <- pca.data.summary$Z / max(abs(pca.data.summary$Z))
pca.data.summary$W <- pca.data.summary$W / max(abs(pca.data.summary$W))


circle <- data.frame( # create a circle for the biplot
  x = cos(seq(0, 2 * pi, length.out = 1000)),
  y = sin(seq(0, 2 * pi, length.out = 1000))
)


# PC1 vs PC2
biplot_pc1v2 <- ggplot() +
  geom_point(data = pca.data.summary, 
             aes(x = X, y = Y), 
             color = "lightblue", alpha = 0.6) +
  geom_segment(data = pca.loadings, 
               aes(x = 0, y = 0, xend = X, yend = Y), 
               arrow = arrow(length = unit(0.2, "cm")),
               color = "#D22B2B") +
  geom_text_repel(data = pca.loadings,
                  aes(x = X, y = Y, label = variable),
                  size          = 3.5,
                  color         = "#D22B2B",
                  box.padding   = 0.2,
                  point.padding = 0.3,
                  segment.size  = 0.3,
                  segment.color = "grey50",
                  max.overlaps  = Inf) +
  geom_path(data = circle, aes(x = x, y = y), 
            color = "lightblue", linetype = "dashed") +
  xlab(paste0("PC1 – ", pca_var_per[1], "%")) +
  ylab(paste0("PC2 – ", pca_var_per[2], "%")) +
  xlim(-1, 1) + ylim(-1, 1) +
  coord_fixed() +
  theme_bw() +
  ggtitle("PCA Biplot (PC1 vs PC2)")

# PC2 vs PC3
biplot_pc2v3 <- ggplot() +
  geom_point(data = pca.data.summary, 
             aes(x = Y, y = Z), 
             color = "lightblue", alpha = 0.6) +
  geom_segment(data = pca.loadings, 
               aes(x = 0, y = 0, xend = Y, yend = Z), 
               arrow = arrow(length = unit(0.2, "cm")),
               color = "#D22B2B") +
  geom_text_repel(data = pca.loadings,
                  aes(x = Y, y = Z, label = variable),
                  size          = 3.5,
                  color         = "#D22B2B",
                  box.padding   = 0.2,
                  point.padding = 0.3,
                  segment.size  = 0.3,
                  segment.color = "grey50",
                  max.overlaps  = Inf) +
  geom_path(data = circle, aes(x = x, y = y), 
            color = "lightblue", linetype = "dashed") +
  xlab(paste0("PC2 – ", pca_var_per[2], "%")) +
  ylab(paste0("PC3 – ", pca_var_per[3], "%")) +
  xlim(-1, 1) + ylim(-1, 1) +
  coord_fixed() +
  theme_bw() +
  ggtitle("PCA Biplot (PC2 vs PC3)")

# PC1 vs PC3
biplot_pc1v3 <- ggplot() +
  geom_point(data = pca.data.summary, 
             aes(x = X, y = Z), 
             color = "lightblue", alpha = 0.6) +
  geom_segment(data = pca.loadings, 
               aes(x = 0, y = 0, xend = X, yend = Z), 
               arrow = arrow(length = unit(0.2, "cm")),
               color = "#D22B2B") +
  geom_text_repel(data = pca.loadings,
                  aes(x = X, y = Z, label = variable),
                  size          = 3.5,
                  color         = "#D22B2B",
                  box.padding   = 0.2,
                  point.padding = 0.3,
                  segment.size  = 0.3,
                  segment.color = "grey50",
                  max.overlaps  = Inf) +
  geom_path(data = circle, aes(x = x, y = y), 
            color = "lightblue", linetype = "dashed") +
  xlab(paste0("PC1 – ", pca_var_per[1], "%")) +
  ylab(paste0("PC3 – ", pca_var_per[3], "%")) +
  xlim(-1, 1) + ylim(-1, 1) +
  coord_fixed() +
  theme_bw() +
  ggtitle("PCA Biplot (PC1 vs PC3)")

# PC1 vs PC4
biplot_pc1v4 <- ggplot() +
  geom_point(data = pca.data.summary, 
             aes(x = X, y = W), 
             color = "lightblue", alpha = 0.6) +
  geom_segment(data = pca.loadings, 
               aes(x = 0, y = 0, xend = X, yend = W), 
               arrow = arrow(length = unit(0.2, "cm")),
               color = "#D22B2B") +
  geom_text_repel(data = pca.loadings,
                  aes(x = X, y = W, label = variable),
                  size          = 3.5,
                  color         = "#D22B2B",
                  box.padding   = 0.2,
                  point.padding = 0.3,
                  segment.size  = 0.3,
                  segment.color = "grey50",
                  max.overlaps  = Inf) +
  geom_path(data = circle, aes(x = x, y = y), 
            color = "lightblue", linetype = "dashed") +
  xlab(paste0("PC1 – ", pca_var_per[1], "%")) +
  ylab(paste0("PC4 – ", pca_var_per[4], "%")) +
  xlim(-1, 1) + ylim(-1, 1) +
  coord_fixed() +
  theme_bw() +
  ggtitle("PCA Biplot (PC1 vs PC4)")

# PC2 vs PC4
biplot_pc2v4 <- ggplot() +
  geom_point(data = pca.data.summary, 
             aes(x = Y, y = W), 
             color = "lightblue", alpha = 0.6) +
  geom_segment(data = pca.loadings, 
               aes(x = 0, y = 0, xend = Y, yend = W), 
               arrow = arrow(length = unit(0.2, "cm")),
               color = "#D22B2B") +
  geom_text_repel(data = pca.loadings,
                  aes(x = Y, y = W, label = variable),
                  size          = 3.5,
                  color         = "#D22B2B",
                  box.padding   = 0.2,
                  point.padding = 0.3,
                  segment.size  = 0.3,
                  segment.color = "grey50",
                  max.overlaps  = Inf) +
  geom_path(data = circle, aes(x = x, y = y), 
            color = "lightblue", linetype = "dashed") +
  xlab(paste0("PC2 – ", pca_var_per[2], "%")) +
  ylab(paste0("PC4 – ", pca_var_per[4], "%")) +
  xlim(-1, 1) + ylim(-1, 1) +
  coord_fixed() +
  theme_bw() +
  ggtitle("PCA Biplot (PC2 vs PC4)")

# PC3 vs PC4
biplot_pc3v4 <- ggplot() +
  geom_point(data = pca.data.summary, 
             aes(x = Z, y = W), 
             color = "lightblue", alpha = 0.6) +
  geom_segment(data = pca.loadings, 
               aes(x = 0, y = 0, xend = Z, yend = W), 
               arrow = arrow(length = unit(0.2, "cm")),
               color = "#D22B2B") +
  geom_text_repel(data = pca.loadings,
                  aes(x = Z, y = W, label = variable),
                  size          = 3.5,
                  color         = "#D22B2B",
                  box.padding   = 0.2,
                  point.padding = 0.3,
                  segment.size  = 0.3,
                  segment.color = "grey50",
                  max.overlaps  = Inf) +
  geom_path(data = circle, aes(x = x, y = y), 
            color = "lightblue", linetype = "dashed") +
  xlab(paste0("PC3 – ", pca_var_per[3], "%")) +
  ylab(paste0("PC4 – ", pca_var_per[4], "%")) +
  xlim(-1, 1) + ylim(-1, 1) +
  coord_fixed() +
  theme_bw() +
  ggtitle("PCA Biplot (PC3 vs PC4)")

grid.arrange(biplot_pc1v2, biplot_pc1v3, ncol = 2)
grid.arrange(biplot_pc1v4,biplot_pc2v3, ncol = 2)
grid.arrange(biplot_pc2v4,biplot_pc3v4, ncol = 2)

# How to interpret loadings: notion sl/unsup/loadings

# The fact that the first component explains more than half of the variance and that
# it is mostly influence by child_mort, income , life expect, total_fer and gdpp, suggests that 
# these variables are the most important in determining the differences between countries.


# Now run clustering on the both the data_scaled_log and the PCA scores
wss <- function(x, k) {
  km <- kmeans(x, centers = k, nstart = 25)
  km$tot.withinss
}

# Try K = 1..10
ks <- 1:10
wss_vals <- sapply(ks, function(k) wss(data_scaled_log, k))

plot(ks, wss_vals, type="b",
     xlab="Number of clusters K",
     ylab="Total within‐cluster SS",
     main="Elbow method for K-means clustering")

pc_scores <- as.data.frame(pca_result$x[, 1:4])  # pick top 4 PCs

wss_pca <- sapply(ks, function(k) wss(pc_scores, k))
lines(ks, wss_pca, type="b", col="blue", lty=2)
legend("topright", legend=c("raw","PCA"), col=c("black","blue"), lty=1:2)


set.seed(123)    # for reproducibility

# 2a) on the scaled+logged data
k <- 3
km_raw <- kmeans(data_scaled_log, centers = k, nstart = 25)
table(km_raw$cluster)

# 2b) on the PCA‐reduced data
km_pca <- kmeans(pc_scores, centers = k, nstart = 25)
table(km_pca$cluster)


# add the assignments back to your original data‐frame
df <- data.frame(data$country, data_scaled_log,
                 cluster_raw = factor(km_raw$cluster), # 
                 cluster_pca = factor(km_pca$cluster))
colnames(df)[1] <- "country" # rename the first column to "country"

# visualize on PC1 vs PC2
ggplot(df, aes(x = pca_result$x[,1], y = pca_result$x[,2], color = cluster_raw)) +
  geom_point(alpha=0.7) + ggtitle("Clusters on raw data")

ggplot(df, aes(x = pca_result$x[,1], y = pca_result$x[,2], color = cluster_pca)) +
  geom_point(alpha=0.7) + ggtitle("Clusters on PCA data")


# 1) Add PC1/PC2 columns to your df
df <- df %>%
  mutate(
    PC1 = pca_result$x[,1],
    PC2 = pca_result$x[,2]
  )

# 2) Pull out % variance explained for each axis
var_per <- round(100 * pca_result$sdev^2 / sum(pca_result$sdev^2), 1)
xlab <- paste0("PC1 (", var_per[1], "%)")
ylab <- paste0("PC2 (", var_per[2], "%)")

# 3) Compute centroids
centroids_raw <- df %>%
  group_by(cluster_raw) %>%
  summarize(PC1 = mean(PC1), PC2 = mean(PC2))

centroids_pca <- df %>%
  group_by(cluster_pca) %>%
  summarize(PC1 = mean(PC1), PC2 = mean(PC2))

# 4) Build panel function
make_panel <- function(cluster_col, centroids, title, pal) {
  ggplot(df, aes(x = PC1, y = PC2, color = .data[[cluster_col]])) +
    geom_point(alpha = 0.8, size = 2) +
    stat_ellipse(aes(fill = .data[[cluster_col]]),
                 type = "norm", level = 0.68, alpha = 0.2, show.legend = FALSE) +
    geom_point(data = centroids, aes(x = PC1, y = PC2),
               shape = 4, size = 4, stroke = 2) +
    geom_text_repel(data = centroids,
                    aes(x = PC1, y = PC2, label = .data[[cluster_col]]),
                    size = 4, color = "black") +
    scale_color_brewer(palette = pal) +
    labs(title = title, x = xlab, y = ylab, color = "Cluster") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# 5) Create the two panels
p_raw <- make_panel(
  cluster_col = "cluster_raw",
  centroids  = centroids_raw,
  title      = "Clusters on raw data",
  pal        = "Set1"
)

p_pca <- make_panel(
  cluster_col = "cluster_pca",
  centroids  = centroids_pca,
  title      = "Clusters on PCA data",
  pal        = "Set2"
)

# 6) Combine them with a shared legend
p_raw + p_pca + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom")





# 1) grab world polygons (as an sf object)
world <- ne_countries(scale = "medium", returnclass = "sf")

# 2) make sure your df has ISO3 codes named "iso_a3"
df$iso_a3 <- countrycode(df$country, origin = "country.name", destination = "iso3c")


# 3) join cluster labels onto the map
world_clust <- world %>% 
  left_join(df, by = c("iso_a3"))



# 4) Plot it 
ggplot(world_clust) +
  geom_sf(aes(fill = as.factor(cluster_raw)), color = "grey80", size = 0.1) +
  scale_fill_manual(
    values = c("1" = "green3", "2" = "blue", "3" = "red"),  # swap red and green
    na.value = "white",
    name = "Cluster"
  ) +
  labs(title = "World Map of Countries by K-means Cluster",
       subtitle = "Clusters computed on raw (scaled+logged) data") +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "transparent"),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )

#
#
#
#
#
# Supervised learning
#
#
#
#
#
#
# let's say I find an index telling how bad the quality of life is in a country. 
# My goal would be to run several models to predict that value (in the whole dataset)
# and then understand which variable influence the most this indicator so that I can 
# then invest in the country with lowest vaues of that/those variables. 

# Check roadmap on notion/sl/sup

# To start the supervised part we need a target variable. At this link:
# "https://openwashdata.github.io/worldhdi/index.html" I found a dataset
# returning the Human Development Index, by the United Nations.

hdi <- read.csv("data/hdi.csv")
hdi <- hdi %>%
  select(hdi_2010, iso3c)
# Now we should add the iso to our datasets, so that we can merge the happiness index:
data$iso_a3 <- countrycode(data$country, origin = "country.name", destination = "iso3c")

# Left join the happiness index to the data
data <- data %>%
  left_join(
   hdi, # rename the column to "hdi"
  by = c("iso_a3" = "iso3c")
  )
# count NAs number of happiness column:
sum(is.na(data$hdi_2010)) # 0 NAs

data_log <- data %>% # create data_log and add 5 to avoid NAs
  select(-iso_a3, -country) %>%
  mutate(across(everything(), ~ log(.x + 5)))


# let's reorder our datasets columns:
new_order <- c("country" , "iso_a3", "child_mort", "exports",  "health",    
"imports","inflation" ,"life_expec","total_fer", "hdi_2010",  "income", "gdpp" )

data <- data %>% select(all_of(new_order))
data_log$country <- data$country
data_log$iso_a3 <- data$iso_a3
data <- data %>% select(all_of(new_order))
data_log <- data_log %>% select(all_of(new_order))

# Now EDA on the dependent variable, HDI:


# Raw HDI
summary(data$hdi_2010)
sd(data$hdi_2010)
IQR(data$hdi_2010)

# Logged HDI
summary(data_log$hdi_2010)
sd(data_log$hdi_2010)
IQR(data_log$hdi_2010)


# Raw HDI
p1 <- ggplot(data, aes(x = hdi_2010)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Histogram of HDI", x = "HDI", y = "Count") +
  theme_minimal()

# Logged HDI
p2 <- ggplot(data_log, aes(x = hdi_2010)) +
  geom_histogram(bins = 30, fill = "darkorange", color = "white") +
  labs(title = "Histogram of Logged HDI", x = "log(HDI + constant)", y = "Count") +
  theme_minimal()

# density plots

# Raw HDI
p3 <- ggplot(data, aes(x = hdi_2010)) +
  geom_density(fill = "skyblue", alpha = 0.5) +
  labs(title = "Density Plot of HDI", x = "HDI") +
  theme_minimal()

# Logged HDI
p4 <- ggplot(data_log, aes(x = hdi_2010)) +
  geom_density(fill = "tomato", alpha = 0.5) +
  labs(title = "Density Plot of Logged HDI", x = "log(HDI + constant)") +
  theme_minimal()

# Box PLots

# Raw HDI
p5 <- ggplot(data, aes(y = hdi_2010)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Boxplot of HDI") +
  theme_minimal()

# Logged HDI
p6 <- ggplot(data_log, aes(y = hdi_2010)) +
  geom_boxplot(fill = "salmon") +
  labs(title = "Boxplot of Logged HDI") +
  theme_minimal()

grid.arrange(p1, p2, p3, p4,p5, p6, ncol = 2)


# Now let's do the Shapiro-Wilk test for normality


# Original
shapiro.test(data$hdi_2010)

# Logged
shapiro.test(data_log$hdi_2010)

# Both test reject hypothesis of normality

# QQ-plot HDI raw
qqnorm(data$hdi_2010); qqline(data$hdi_2010, col = "red")

# QQ-plot HDI log
qqnorm(data_log$hdi_2010); qqline(data_log$hdi_2010, col = "blue")


# The QQ plot shows some normality whereas the Shapiro Wilk does not. 
# We will tackle both the approach of using robust methods that do not
# require normality and the approach of models like linear regression
# that do not actually require normality on the target variable, but on the residuals.

# Our goal is to understand the most influential variables on the hdi,
# check sl/sup/variable_importance for more details

data_scaled <- as.data.frame(scale(data[3:10])) # scale the data to have mean = 0 and sd = 1 and drop income and gdpp 
# Let's start with the linear regression

# Linear regression on the original data
# We should first exclude the multicollinearity variables: income and gdpp

lm_model <- lm(hdi_2010 ~ ., data = data_scaled) #until 10 to exclude 11: income and 12:gdpp
summary(lm_model) # R uses the hdi_2010 inside data[3:10] as the target
# The . operator means: "use all other columns in data[3:10] as predictors"
 
# plot residuals
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0))  # Larger top outer margin

# Plot diagnostics with base R (includes leverage plot)
plot(lm_model, which = 1:4, col = "blue", pch = 19, cex = 0.7,sub.caption = "")

# Add a global title, slightly lower than default to avoid overlap
mtext("Diagnostic Plots for Linear Model", outer = TRUE, cex = 1.4, font = 2, line = 1)


# the scale-location suggest slight heteroskedasticity, but doesn't seem anything concerning
# the Cook's distance suggests that the following:
data[c(67, 114, 152), ]
# are influential observations
#
#
# same regression and residual analysis but with the logged data
#
#
data_log_scaled <- as.data.frame(scale(data_log[3:10])) # drop income and gdpp 

lm_model <- lm(hdi_2010 ~ ., data = data_log_scaled) #exclude income and gdpp
summary(lm_model)

# plot residuals
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0))  # Larger top outer margin

# Plot diagnostics with base R (includes leverage plot)
plot(lm_model, which = 1:4, col = "blue", pch = 19, cex = 0.7,sub.caption = "")

# Add a global title, slightly lower than default to avoid overlap
mtext("Diagnostic Plots for Linear Model", outer = TRUE, cex = 1.4, font = 2, line = 1)

# the scale-location suggest slight heteroskedasticity, but doesn't seem anything concerning
# the Cook's distance suggests that the following:
data[c(67, 150, 155), ]
# are influential observations


# NOW LET'S GO WITH LASSO PREDICTION
# Explicitly remove the target from the predictors
#thorough explanation of the following code: notion/sl/sup/lasso_theory_recap/thorough_explanation

# --- 1. Prepare data ---
x_raw <- data %>%
  select(-hdi_2010, -iso_a3, -country) %>%
  as.matrix()

y_raw <- data$hdi_2010

# --- 2. Run LASSO with cross-validation ---
cv_lasso_raw <- cv.glmnet(x_raw, y_raw, alpha = 1, standardize = TRUE) # alpha = 1 for LASSO
# Uses k-fold cross-validation (default is 10-fold) to select the best value 
# of λ (lambda), the penalty parameter.


# --- 3. Create tidy dataframe for ggplot of CV error ---
lasso_plot_data <- data.frame(
  lambda = cv_lasso_raw$lambda,
  mse = cv_lasso_raw$cvm,
  lower = cv_lasso_raw$cvlo,
  upper = cv_lasso_raw$cvup
) %>%
  mutate(
    label = case_when(
      abs(lambda - cv_lasso_raw$lambda.min) < 1e-10 ~ "min", 
      # `cv_lasso_raw$lambda.min`: the value of lambda that gives the 
      # lowest mean squared error (MSE) in cross-validation.
      
      abs(lambda - cv_lasso_raw$lambda.1se) < 1e-10 ~ "1se",
      TRUE ~ "none"
      #`cv_lasso_raw$lambda.1se`: a more conservative lambda, 
      # within 1 standard error of the best — often used for a simpler model.
    )
  )

# --- 4. Plot cross-validation error ---
ggplot(lasso_plot_data, aes(x = log(lambda), y = mse)) +
  geom_line(color = "steelblue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "lightblue") +
  geom_point(aes(color = label), size = 2) +
  geom_vline(xintercept = log(cv_lasso_raw$lambda.min), linetype = "dashed", color = "red") +
  geom_vline(xintercept = log(cv_lasso_raw$lambda.1se), linetype = "dotted", color = "darkgreen") +
  scale_color_manual(values = c("min" = "red", "1se" = "darkgreen", "none" = "grey")) +
  labs(
    title = "LASSO Cross-Validation: Mean Squared Error vs log(Lambda)",
    x = "log(Lambda)",
    y = "Mean Squared Error (CV)",
    color = "Lambda"
  ) +
  theme_minimal()

# --- 5. Extract and tidy coefficients at lambda.min ---
lasso_coefs <- coef(cv_lasso_raw, s = "lambda.min")

# Convert sparse matrix to data frame safely
lasso_coefs_df <- lasso_coefs %>%
  as.matrix() %>%
  data.frame() %>%
  tibble::rownames_to_column(var = "variable")

# Rename column safely (in case of unnamed column)
names(lasso_coefs_df)[2] <- "coefficient"

# --- 6. Plot coefficients if any non-zero (excluding intercept) ---
lasso_coefs_df <- lasso_coefs_df %>%
  filter(variable != "(Intercept)", coefficient != 0)

if (nrow(lasso_coefs_df) == 0) {
  message("⚠️ No variables were selected by LASSO at lambda.min.")
} else {
  lasso_coefs_df <- lasso_coefs_df %>%
    mutate(abs_coef = abs(coefficient)) %>%
    arrange(desc(abs_coef))
  
  ggplot(lasso_coefs_df, aes(x = reorder(variable, coefficient), y = coefficient, fill = coefficient > 0)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato")) +
    labs(
      title = "LASSO-Selected Variables (No Intercept)",
      subtitle = "Non-zero coefficients at lambda.min",
      x = "Variable",
      y = "Coefficient",
      fill = "Direction"
    ) +
    theme_minimal()
}


# SAME WITH LOGGED DATA

# --- 1. Prepare data ---
x_log <- data_log %>%
  select(-hdi_2010, -iso_a3, -country) %>%
  as.matrix()

y_log <- data_log$hdi_2010

# --- 2. Run LASSO with cross-validation ---
cv_lasso_log <- cv.glmnet(x_log, y_log, alpha = 1, standardize = TRUE) 
# alpha = 1 for LASSO (L1 regularization)

# --- 3. Create tidy dataframe for ggplot of CV error ---
lasso_plot_data_log <- data.frame(
  lambda = cv_lasso_log$lambda,
  mse = cv_lasso_log$cvm,
  lower = cv_lasso_log$cvlo,
  upper = cv_lasso_log$cvup
) %>%
  mutate(
    label = case_when(
      abs(lambda - cv_lasso_log$lambda.min) < 1e-10 ~ "min",
      abs(lambda - cv_lasso_log$lambda.1se) < 1e-10 ~ "1se",
      TRUE ~ "none"
    )
  )

# --- 4. Plot cross-validation error ---
ggplot(lasso_plot_data_log, aes(x = log(lambda), y = mse)) +
  geom_line(color = "steelblue") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "lightblue") +
  geom_point(aes(color = label), size = 2) +
  geom_vline(xintercept = log(cv_lasso_log$lambda.min), linetype = "dashed", color = "red") +
  geom_vline(xintercept = log(cv_lasso_log$lambda.1se), linetype = "dotted", color = "darkgreen") +
  scale_color_manual(values = c("min" = "red", "1se" = "darkgreen", "none" = "grey")) +
  labs(
    title = "LASSO Cross-Validation (Log Data): MSE vs log(Lambda)",
    x = "log(Lambda)",
    y = "Mean Squared Error (CV)",
    color = "Lambda"
  ) +
  theme_minimal()
# --- 5. Extract and tidy coefficients at lambda.min ---
lasso_coefs_log <- coef(cv_lasso_log, s = "lambda.min")

# Convert sparse matrix to data frame 
lasso_coefs_df_log <- lasso_coefs_log %>%
  as.matrix() %>%
  data.frame() %>%
  tibble::rownames_to_column(var = "variable")

# Rename column 
names(lasso_coefs_df_log)[2] <- "coefficient"

# --- 6. Plot coefficients if any non-zero (excluding intercept) ---
lasso_coefs_df_log <- lasso_coefs_df_log %>%
  filter(variable != "(Intercept)", coefficient != 0)

if (nrow(lasso_coefs_df_log) == 0) {
  message("⚠️ No variables were selected by LASSO at lambda.min.")
} else {
  lasso_coefs_df_log <- lasso_coefs_df_log %>%
    mutate(abs_coef = abs(coefficient)) %>%
    arrange(desc(abs_coef))
  
  ggplot(lasso_coefs_df_log, aes(x = reorder(variable, coefficient), y = coefficient, fill = coefficient > 0)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato")) +
    labs(
      title = "LASSO-Selected Variables (Log Data, No Intercept)",
      subtitle = "Non-zero coefficients at lambda.min",
      x = "Variable",
      y = "Coefficient",
      fill = "Direction (TRUE = Positive)"
    ) +
    theme_minimal()
}


# 1. Prepare data
#    Remove non‐predictors (ISO code, country name) and the target from the predictors,
#    but keep them in the data frame for formula interface.
df_raw <- data %>%
  select(-iso_a3, -country)    # hdi_2010 stays, for formula

# 2. Fit a full (overgrown) regression tree
#    method = "anova" for continuous response,
#    control: cp=0.0001 to let it grow deep, xval=10 for 10‐fold CV.
tree_raw <- rpart(
  formula = hdi_2010 ~ ., 
  data    = df_raw,
  method  = "anova",
  control = rpart.control(cp = 1e-4, xval = 10)
)

# 3. Examine the complexity‐parameter table
#    tree_raw$cptable is a matrix with:
#      CP: complexity parameter,
#      nsplit: number of splits,
#      rel error: training error relative to root,
#      xerror: CV error relative to root,
#      xstd: std. error of CV.
cp_table_raw <- as_tibble(tree_raw$cptable, rownames = "row") %>%
  rename(
    cp       = CP,
    n_splits = nsplit,
    rel_err  = `rel error`,
    x_err    = xerror,
    x_std    = xstd
  )

# 4. Plot cross‐validation error vs complexity (cp)
ggplot(cp_table_raw, aes(x = log(cp), y = x_err)) +
  geom_line(color = "steelblue") +
  geom_ribbon(aes(ymin = x_err - x_std, ymax = x_err + x_std),
              alpha = 0.2, fill = "lightblue") +
  geom_point(size = 2) +
  geom_vline(xintercept = log(cp_table_raw$cp[which.min(cp_table_raw$x_err)]),
             linetype = "dashed", color = "red") +
  labs(
    title = "Tree CV Error vs log(CP) (raw data)",
    x     = "log(CP)",
    y     = "Relative CV Error"
  ) +
  theme_minimal()

# 5. Choose the optimal CP
#    cp_min: gives lowest CV error;
#    cp_1se: the largest cp within 1 std‐error of the minimum (more conservative).
min_row   <- which.min(cp_table_raw$x_err)
cp_min    <- cp_table_raw$cp[min_row]
cp_1se    <- cp_table_raw$cp[ max(which(cp_table_raw$x_err <= cp_table_raw$x_err[min_row] + cp_table_raw$x_std[min_row])) ]

# 6. Prune the tree at cp_min (or cp_1se, if you prefer simpler model)
pruned_raw <- prune(tree_raw, cp = cp_min)

# visualize the pruned tree
rpart.plot(pruned_raw, main = sprintf("Pruned Regression Tree (cp = %.5f)", cp_min))

# 7. Extract variable importance
#    rpart stores a named vector in $variable.importance
varimp_raw <- pruned_raw$variable.importance %>%
  enframe(name = "variable", value = "importance") %>%
  arrange(desc(importance))

# 8. Plot variable importance as a bar chart
ggplot(varimp_raw, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Variable Importance (Pruned Tree, raw data)",
    x     = "Variable",
    y     = "Relative Importance"
  ) +
  theme_minimal()



# Regression‐Tree Prediction (logged data) ---------------------------------

# 1. Prepare logged data
df_log <- data_log %>%
  select(-iso_a3, -country)   # hdi_2010 stays

# 2. Fit full tree on logged data
tree_log <- rpart(
  hdi_2010 ~ .,
  data    = df_log,
  method  = "anova",
  control = rpart.control(cp = 1e-4, xval = 10)
)

# 3. CP table & tibble
cp_table_log <- as_tibble(tree_log$cptable, rownames = "row") %>%
  rename(cp = CP, n_splits = nsplit, rel_err = `rel error`, x_err = xerror, x_std = xstd)

# 4. Plot CV error vs log(cp)
ggplot(cp_table_log, aes(x = log(cp), y = x_err)) +
  geom_line(color = "steelblue") +
  geom_ribbon(aes(ymin = x_err - x_std, ymax = x_err + x_std),
              alpha = 0.2, fill = "lightblue") +
  geom_point(size = 2) +
  geom_vline(xintercept = log(cp_table_log$cp[which.min(cp_table_log$x_err)]),
             linetype = "dashed", color = "red") +
  labs(
    title = "Tree CV Error vs log(CP) (log data)",
    x     = "log(CP)",
    y     = "Relative CV Error"
  ) +
  theme_minimal()

# 5. Find cp_min, cp_1se for logged data
min_row_log <- which.min(cp_table_log$x_err)
cp_min_log  <- cp_table_log$cp[min_row_log]
cp_1se_log  <- cp_table_log$cp[
  max(which(cp_table_log$x_err <= cp_table_log$x_err[min_row_log] + cp_table_log$x_std[min_row_log]))
]

# 6. Prune at cp_min_log
pruned_log <- prune(tree_log, cp = cp_min_log)
rpart.plot(pruned_log, main = sprintf("Pruned Regression Tree (cp = %.5f) — logged data", cp_min_log))

# 7. Variable importance (logged)
varimp_log <- pruned_log$variable.importance %>%
  enframe(name = "variable", value = "importance") %>%
  arrange(desc(importance))

# 8. Plot importance
ggplot(varimp_log, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Variable Importance (Pruned Tree, log data)",
    x     = "Variable",
    y     = "Relative Importance"
  ) +
  theme_minimal()


# 1. Prepare the data (raw)
df_raw <- data %>% select(-iso_a3, -country)

# 2. (Optional) Tune mtry
# tune <- tuneRF(
#   x = df_raw %>% select(-hdi_2010),
#   y = df_raw$hdi_2010,
#   ntreeTry = 200,
#   stepFactor = 1.5,
#   improve = 0.01,
#   trace = TRUE
# )
# best_mtry <- tune[which.min(tune[, “OOBError” ]), “mtry” ]

# 3. Fit the Random Forest
set.seed(42)
rf_raw <- randomForest(
  hdi_2010 ~ .,
  data      = df_raw,
  ntree     = 500,
  mtry      = floor(ncol(df_raw)/3),  # or use best_mtry from above
  importance = TRUE
)

# 4. Extract and plot importance
varimp_raw <- as.data.frame(importance(rf_raw)) %>%
  rownames_to_column("variable") %>%
  rename(
    IncMSE         = `%IncMSE`,
    IncNodePurity  = `IncNodePurity`
  ) %>%
  arrange(desc(IncMSE))

ggplot(varimp_raw, aes(x = reorder(variable, IncMSE), y = IncMSE)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Random Forest Variable Importance (raw data)",
    x     = "Variable",
    y     = "IncMSE"
  ) +
  theme_minimal()

# 5. Repeat on log‐transformed data
df_log <- data_log %>% select(-iso_a3, -country)
set.seed(42)
rf_log <- randomForest(
  hdi_2010 ~ .,
  data       = df_log,
  ntree      = 500,
  mtry       = floor(ncol(df_log)/3),
  importance = TRUE
)

varimp_log <- as.data.frame(importance(rf_log)) %>%
  rownames_to_column("variable") %>%
  rename(IncMSE = `%IncMSE`) %>%
  arrange(desc(IncMSE))

ggplot(varimp_log, aes(x = reorder(variable, IncMSE), y = IncMSE)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Random Forest Variable Importance (logged data)",
    x     = "Variable",
    y     = "IncMSE"
  ) +
  theme_minimal()

