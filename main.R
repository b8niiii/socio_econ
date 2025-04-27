library(corrplot)
library(car)
library(Hmisc) # for redundancy analysis
library(ggplot2)
library(gridExtra)
library(dplyr)

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


# standardize data

data_scaled <- as.data.frame(scale(data[2:10]))
data_scaled_log <- as.data.frame(scale(data_log))
 

# let's run redundancy analysis on the scaled log data

redun_result_scaled <- redun(~ ., data = data_scaled_log, nk = 0)
print(redun_result_scaled)

# we got the same results as before 

# Let's do the PCA on the scaled log data now:

pca_log <- prcomp(data_scaled_log)
summary(pca_result)
# Plot PCA results
par(mfrow = c(1, 2)) # Set up a 1x2 grid for plots
plot(pca_result, main = "PCA Scree Plot") # Scree plot
biplot(pca_result, main = "PCA Biplot") # Biplot

# PCA from old project

pca_log_var <- pca_log$sdev^2 # variance of each principal component
pca_var_per <- round(pca_log_var/sum(pca_log_var)*100, 2) # percentage of variance explained by each component

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

par(mfrow = c(1,1)) 
rownames(pca_log$x) <-1:nrow(pca_log$x) # Set row names to be the row numbers
pca_data <-as.data.frame(pca_log$x)
pca.data.summary <- data.frame(sample=rownames(pca_log$x),
                               X=pca_log$x[,1],
                               Y=pca_log$x[,2],
                               Z=pca_log$x[,3],
                               row.names=NULL)

pca.loadings <- data.frame(variable=names(data_log),
                           X=pca_log$rotation[,1],
                           Y=pca_log$rotation[,2],
                           Z=pca_log$rotation[,3],
                           row.names=NULL)

#copy-pasted from the project with Ali:
# Normalize sample scores to fit within [-1,1] for better visualization
pca.data.summary$X <- pca.data.summary$X / max(abs(pca.data.summary$X))
pca.data.summary$Y <- pca.data.summary$Y / max(abs(pca.data.summary$Y))
pca.data.summary$Z <- pca.data.summary$Z / max(abs(pca.data.summary$Z))

circle <- data.frame(
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

grid.arrange(biplot_pc1v2, biplot_pc2v3, biplot_pc1v3, ncol = 3)

# PCA 3D plot (PC 1,2,3)
pca_log_scaled <- pca_log$x / max(abs(pca_log$x))
pca.loadings.scaled <- pca_log$rotation / max(abs(pca_log$rotation))

pca_log_plot_threeD <- plot_ly() %>%
  add_trace(x = pca_log_scaled[,1], 
            y = pca_log_scaled[,2], 
            z = pca_log_scaled[,3], 
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
            text = rownames(pca_log$rotation),
            textposition = 'top center',
            textfont = list(color = '#D22B2B', size = 12),
            name = 'Variables') %>%
  layout(title = '3D PCA Graph',
         scene = list(xaxis = list(title = paste('PC1 - ', pca_var_per[1], '%')),
                      yaxis = list(title = paste('PC2 - ', pca_var_per[2], '%')),
                      zaxis = list(title = paste('PC3 - ', pca_var_per[3], '%'))))




