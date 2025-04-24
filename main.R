data <- Country.data
summary(data)
# Histograms, boxplot: check variables' distributions
par(mfrow=c(6, 4), mar=c(2, 2, 2, 2), oma=c(0, 0, 3, 0))
for(i in 2:23) {
  hist(data[, i], 
       main=names(data)[i], 
       col="lightblue")}
mtext("Variables original distribution", outer=TRUE, cex=1.5, font=2)
dev.off()
