library(CausalImpact)

set.seed(1)
x1 <- 100 + arima.sim(model = list(ar = 0.999), n = 100)
x2 <- 100 + arima.sim(model = list(ar = 0.999), n = 100)
x3 <- 100 + arima.sim(model = list(ar = 0.999), n = 100)
x4 <- 100 + arima.sim(model = list(ar = 0.999), n = 100)
y <- 1.2 * x1 + rnorm(100) + .8*x2 + .7*x3 + .6*x4
y[71:100] <- y[71:100] + 10
data <- cbind(y, x1, x2, x3, x4)

pre.period <- c(1, 3) #70
post.period <- c(4, 100)

#impact <- CausalImpact(data, pre.period, post.period, alpha = 0.05)
impact <- CausalImpact(data, pre.period, post.period)

plot(impact)
summary(impact)

plot(impact$model$bsts.model, "coefficients")
impact$summary$AbsEffect
impact$summary$AbsEffect.lower
impact$summary$AbsEffect.upper
s = impact$series

summary(lm(s$point.pred~x1+x2))
