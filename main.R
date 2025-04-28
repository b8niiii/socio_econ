''' https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data

Problem Statement:
  HELP International have been able to raise around $ 10 million. 
Now the CEO of the NGO needs to decide how to use this money strategically 
and effectively. So, CEO has to make decision to choose the countries that are 
in the direst need of aid. Hence, your Job as a Data scientist is to categorise
the countries using some socio-economic and health factors that determine the 
overall development of the country. Then you need to suggest the countries 
which the CEO needs to focus on the most. '''


library(corrplot)
library(car)
library(Hmisc) # for redundancy analysis
library(ggplot2)
library(gridExtra)
library(dplyr)
library(plotly)
library(glmnet) # for ridge regression
library(reshape2) # for reshaping the data easily

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
data_log <- na.omit(data_log) # Drop rows with NA values
par(mfrow=c(3, 3), mar=c(2, 2, 2, 2), oma=c(0, 0, 3, 0)) # 3 rows, 3 columns
for(i in 2:9) {
  hist(data_log[,i], 
          main = names(data)[i], 
          col = "lightblue")}
mtext("Log Variables Histograms", outer=TRUE, cex=1.0, font=2)


par(mfrow=c(3, 3), mar=c(2, 2, 2, 2), oma=c(0, 0, 3, 0)) # 3 rows, 3 columns
for(i in 2:9) {
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
# AND IN GENERAL HOW THERE WERE NONLINEAR CORRELATION NOT SEEN BY THE CORRELATION MATRIX WITH 
# NORMAL VARIABLES AND THE VIF, BUT EASILY SPOTTED WITH THE CORRELATION MATRIX AND THE VIF
# ON LOGARITHMIC VARIABLES, CAUSE THESE RELATIONS ARE NOT LINEAR AND THE LOG LINEARIZES THEM


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

#Redundancy analysis on original variables did not find redundant features because 
#relationships were nonlinear. After log transformation, "income" and "child mortality" 
#became linearly predictable from other variables. This shows that the log transformation 
#revealed hidden dependencies that were not detectable before.


# redundancy analysis on original data
redun_result <- redun(~ ., data = data[2:10], nk = 0) 
print(redun_result)


# redundancy analysis on log
redun_result <- redun(~ ., data = data_log, nk = 0) 
print(redun_result)

# LET'S TRY TO GET A LIL BIT DEEPER INTO THE CORRELATIONS AMONG INCOME, GDPP AND CHILD MORT
# WE SPOTTED THOSE PROBLEMATIC VARIABLES DOING THE CORRELATION MATRIX AND ANALYZING WITH
# THE VIF AND THE REDUNDANCY ANALYSIS.

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
  labs(title = "Income vs GDP per capita", x = "GDP per capita", y = "Income") +
  theme_minimal()

ggplot(data_scaled_log, aes(x = gdpp, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_smooth(method = "lm", color = 'red') +
  labs(title = "Log-Log Relationship", x = "log(GDP per capita)", y = "log(Income)") +
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

# Then plot
ggplot(data, aes(x = gdpp, y = income)) +
  geom_point(color = 'blue', alpha = 0.6) +
  stat_function(fun = function(x) exp(alpha) * x^beta, color = 'red', size = 1.2) +
  labs(
    title = "Income vs GDP per capita (Original Scale with Correct Power-Law Fit)",
    x = "GDP per capita",
    y = "Income"
  ) +
  theme_minimal()

# not satisfied with the power law fit, let's try to fit a polynomial regression
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

# now to officially say that these two are not linearly correlated, I will run a linear model
# and compare the adjusted R^2:
linear_model <- lm(income ~ gdpp, data = data)
summary(linear_model)
summary(poly2_model)

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
  labs(title = "Log-Log Relationship", x = "log(Child Mortality)", y = "log(Income)") +
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

# Compare linear vs quadratic model
linear_model <- lm(income ~ child_mort, data = data)
summary(linear_model)
summary(poly2_model)
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

# Create a data frame summarizing feature loadings on the first 3 principal components
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

# PC1 VS PC2
biplot_pc1v2 <- ggplot() +
  geom_point(data = pca.data.summary, aes(x = X, y = Y), color = 'lightblue', alpha = 0.6) +  
  geom_segment(data = pca.loadings, aes(x = 0, y = 0, xend = X, yend = Y), 
               arrow = arrow(length = unit(0.2, 'cm')), color = '#D22B2B') +
  geom_text(data = pca.loadings, aes(x = X, y = Y, label = variable), 
            size = 3.5, color = '#D22B2B', vjust = 1.5) +  
  geom_path(data = circle, aes(x = x, y = y), color = 'lightblue', linetype = 'dashed') +
  xlab(paste('PC1 - ', pca_var_per[1], '%', sep = '')) +
  ylab(paste('PC2 - ', pca_var_per[2], '%', sep = '')) +
  xlim(-1, 1) + ylim(-1, 1) +  
  coord_fixed() +
  theme_bw() +
  ggtitle('PCA Biplot (PC1 vs PC2)')

# PC2 VS PC3
biplot_pc2v3 <- ggplot() +
  geom_point(data = pca.data.summary, aes(x = Y, y = Z), color = 'lightblue', alpha = 0.6) +  
  geom_segment(data = pca.loadings, aes(x = 0, y = 0, xend = Y, yend = Z), 
               arrow = arrow(length = unit(0.2, 'cm')), color = '#D22B2B') +
  geom_text(data = pca.loadings, aes(x = Y, y = Z, label = variable), 
            size = 3.5, color = '#D22B2B', vjust = 1.5) +  
  geom_path(data = circle, aes(x = x, y = y), color = 'lightblue', linetype = 'dashed') +
  xlab(paste('PC2 - ', pca_var_per[2], '%', sep = '')) +
  ylab(paste('PC3 - ', pca_var_per[3], '%', sep = '')) +
  xlim(-1, 1) + ylim(-1, 1) +  
  coord_fixed() +
  theme_bw() +
  ggtitle('PCA Biplot (PC2 vs PC3)')

# PC1 VS PC3
biplot_pc1v3 <- ggplot() +
  geom_point(data = pca.data.summary, aes(x = X, y = Z), color = 'lightblue', alpha = 0.6) +  
  geom_segment(data = pca.loadings, aes(x = 0, y = 0, xend = X, yend = Z), 
               arrow = arrow(length = unit(0.2, 'cm')), color = '#D22B2B') +
  geom_text(data = pca.loadings, aes(x = X, y = Z, label = variable), 
            size = 3.5, color = '#D22B2B', vjust = 1.5) +  
  geom_path(data = circle, aes(x = x, y = y), color = 'lightblue', linetype = 'dashed') +
  xlab(paste('PC1 - ', pca_var_per[1], '%', sep = '')) +
  ylab(paste('PC3 - ', pca_var_per[3], '%', sep = '')) +
  xlim(-1, 1) + ylim(-1, 1) +  
  coord_fixed() +
  theme_bw() +
  ggtitle('PCA Biplot (PC1 vs PC3)')

# PC1 vs PC4
biplot_pc1v4 <- ggplot() +
  geom_point(data = pca.data.summary, aes(x = X, y = W), color = 'lightblue', alpha = 0.6) +  
  geom_segment(data = pca.loadings, aes(x = 0, y = 0, xend = X, yend = W), 
               arrow = arrow(length = unit(0.2, 'cm')), color = '#D22B2B') +
  geom_text(data = pca.loadings, aes(x = X, y = W, label = variable), 
            size = 3.5, color = '#D22B2B', vjust = 1.5) +  
  geom_path(data = circle, aes(x = x, y = y), color = 'lightblue', linetype = 'dashed') +
  xlab(paste('PC1 - ', pca_var_per[1], '%', sep = '')) +
  ylab(paste('PC4 - ', pca_var_per[4], '%', sep = '')) +
  xlim(-1, 1) + ylim(-1, 1) +  
  coord_fixed() +
  theme_bw() +
  ggtitle('PCA Biplot (PC1 vs PC4)')

# PC2 vs PC4
biplot_pc2v4 <- ggplot() +
  geom_point(data = pca.data.summary, aes(x = Y, y = W), color = 'lightblue', alpha = 0.6) +  
  geom_segment(data = pca.loadings, aes(x = 0, y = 0, xend = Y, yend = W), 
               arrow = arrow(length = unit(0.2, 'cm')), color = '#D22B2B') +
  geom_text(data = pca.loadings, aes(x = Y, y = W, label = variable), 
            size = 3.5, color = '#D22B2B', vjust = 1.5) +  
  geom_path(data = circle, aes(x = x, y = y), color = 'lightblue', linetype = 'dashed') +
  xlab(paste('PC2 - ', pca_var_per[2], '%', sep = '')) +
  ylab(paste('PC4 - ', pca_var_per[4], '%', sep = '')) +
  xlim(-1, 1) + ylim(-1, 1) +  
  coord_fixed() +
  theme_bw() +
  ggtitle('PCA Biplot (PC2 vs PC4)')

# PC3 vs PC4
biplot_pc3v4 <- ggplot() +
  geom_point(data = pca.data.summary, aes(x = Z, y = W), color = 'lightblue', alpha = 0.6) +  
  geom_segment(data = pca.loadings, aes(x = 0, y = 0, xend = Z, yend = W), 
               arrow = arrow(length = unit(0.2, 'cm')), color = '#D22B2B') +
  geom_text(data = pca.loadings, aes(x = Z, y = W, label = variable), 
            size = 3.5, color = '#D22B2B', vjust = 1.5) +  
  geom_path(data = circle, aes(x = x, y = y), color = 'lightblue', linetype = 'dashed') +
  xlab(paste('PC3 - ', pca_var_per[3], '%', sep = '')) +
  ylab(paste('PC4 - ', pca_var_per[4], '%', sep = '')) +
  xlim(-1, 1) + ylim(-1, 1) +  
  coord_fixed() +
  theme_bw() +
  ggtitle('PCA Biplot (PC3 vs PC4)')

grid.arrange(biplot_pc1v2, biplot_pc2v3, biplot_pc1v3,biplot_pc1v4,biplot_pc2v4,biplot_pc3v4, ncol = 3)

# PCA 3D plot (PC 1,2,3)
pca_result_scaled <- pca_result$x / max(abs(pca_result$x))
pca.loadings.scaled <- pca_result$rotation / max(abs(pca_result$rotation))

pca_result_plot_threeD <- plot_ly() %>%
  add_trace(x = pca_result_scaled[,1], 
            y = pca_result_scaled[,2], 
            z = pca_result_scaled[,3], 
            type = 'scatter3d', mode = 'markers',
            marker = list(color = 'lightblue', size = 3),
            name = 'Samples') %>%
  add_trace(x = rep(0, nrow(pca.loadings.scaled)), 
            y = rep(0, nrow(pca.loadings.scaled)), 
            z = rep(0, nrow(pca.loadings.scaled)),
            xend = pca.loadings.scaled[,1], 
            yend = pca.loadings.scaled[,2], 
            zend = pca.loadings.scaled[,3], 
            type = 'scatter3d', mode = 'lines',
            line = list(color = '#D22B2B', width = 10),
            name = 'PC Loadings') %>%
  add_trace(x = pca.loadings.scaled[,1], 
            y = pca.loadings.scaled[,2], 
            z = pca.loadings.scaled[,3], 
            type = 'scatter3d', mode = 'text',
            text = rownames(pca_result$rotation),
            textposition = 'top center',
            textfont = list(color = '#D22B2B', size = 12),
            name = 'Variables') %>%
  layout(title = '3D PCA Graph',
         scene = list(xaxis = list(title = paste('PC1 - ', pca_var_per[1], '%')),
                      yaxis = list(title = paste('PC2 - ', pca_var_per[2], '%')),
                      zaxis = list(title = paste('PC3 - ', pca_var_per[3], '%'))))
pca_result_plot_threeD



