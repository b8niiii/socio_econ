library(corrplot)



data <- Country.data

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
mtext("Variables original distribution", outer=TRUE, cex=1.0, font=2)
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
data_log <- lapply(data[2:10], function(x) log(x + 1))

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


# so the only normal variable seems to be the logarithm of health spending per capita as a percentage of gdp
# even tho the logarithms of the variables turned out not to be normal, we can still plot the results
# to see if they look at least more normal than before:
data_log <- as.data.frame(data_log) # Convert the list of log-transformed data back to a data frame
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

