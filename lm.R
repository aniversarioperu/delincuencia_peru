library(ggplot2)
y <- read.csv("datos.txt", header=FALSE)
y <- as.vector(y[,1])
x <- 2005:2012

int <- lsfit(x,y)$coefficients[1]
slope <- lsfit(x,y)$coefficients[2]
p <- ggplot(,aes(x,y))
p + geom_point() + geom_abline(intercept=int, slope=slope) +
  labs(title = "Perú: Número total de delitos por año")


summary(lm(x ~ y))

# R2 = 0.67
# p-value = 0.008