#
# demonstrate capability of using non-liner model
#
#
# clear environment
#
  rm(list = ls())
#
# clear the console
#
  cat("\014")
#
# copy of neuralnet code
# note source is https://github.com/bips-hb/neuralnet/blob/master/R/neuralnet.r  
#
neuralnet <-
  function(formula, data, hidden = 1, threshold = 0.01, stepmax = 1e+05, 
            rep = 1, startweights = NULL, learningrate.limit = NULL, 
            learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
            lifesign = "none", lifesign.step = 1000, algorithm = "rprop+", 
            err.fct = "sse", act.fct = "logistic", linear.output = TRUE, 
            exclude = NULL, constant.weights = NULL, likelihood = FALSE) 
  {
    call <- match.call()
    options(scipen = 100, digits = 10)
    result <- varify.variables(data, formula, startweights, learningrate.limit, 
                               learningrate.factor, learningrate, lifesign, algorithm, 
                               threshold, lifesign.step, hidden, rep, stepmax, err.fct, 
                               act.fct)
    data <- result$data
    formula <- result$formula
    startweights <- result$startweights
    learningrate.limit <- result$learningrate.limit
    learningrate.factor <- result$learningrate.factor
    learningrate.bp <- result$learningrate.bp
    lifesign <- result$lifesign
    algorithm <- result$algorithm
    threshold <- result$threshold
    lifesign.step <- result$lifesign.step
    hidden <- result$hidden
    rep <- result$rep
    stepmax <- result$stepmax
    model.list <- result$model.list
    matrix <- NULL
    list.result <- NULL
    result <- generate.initial.variables(data, model.list, hidden, 
                                         act.fct, err.fct, algorithm, linear.output, formula)
    covariate <- result$covariate
    response <- result$response
    err.fct <- result$err.fct
    err.deriv.fct <- result$err.deriv.fct
    act.fct <- result$act.fct
    act.deriv.fct <- result$act.deriv.fct
    for (i in 1:rep) {
      if (lifesign != "none") {
        lifesign <- display(hidden, threshold, rep, i, lifesign)
      }
      utils::flush.console()
      result <- calculate.neuralnet(learningrate.limit = learningrate.limit, 
                                    learningrate.factor = learningrate.factor, covariate = covariate, 
                                    response = response, data = data, model.list = model.list, 
                                    threshold = threshold, lifesign.step = lifesign.step, 
                                    stepmax = stepmax, hidden = hidden, lifesign = lifesign, 
                                    startweights = startweights, algorithm = algorithm, 
                                    err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                                    act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                                    rep = i, linear.output = linear.output, exclude = exclude, 
                                    constant.weights = constant.weights, likelihood = likelihood, 
                                    learningrate.bp = learningrate.bp)
      if (!is.null(result$output.vector)) {
        list.result <- c(list.result, list(result))
        matrix <- cbind(matrix, result$output.vector)
      }
    }
    utils::flush.console()
    if (!is.null(matrix)) {
      weight.count <- length(unlist(list.result[[1]]$weights)) - 
        length(exclude) + length(constant.weights) - sum(constant.weights == 
                                                           0)
      if (!is.null(startweights) && length(startweights) < 
          (rep * weight.count)) {
        warning("some weights were randomly generated, because 'startweights' did not contain enough values", 
                call. = F)
      }
      ncol.matrix <- ncol(matrix)
    }
    else ncol.matrix <- 0
    if (ncol.matrix < rep) 
      warning(sprintf("algorithm did not converge in %s of %s repetition(s) within the stepmax", 
                      (rep - ncol.matrix), rep), call. = FALSE)
    nn <- generate.output(covariate, call, rep, threshold, matrix, 
                          startweights, model.list, response, err.fct, act.fct, 
                          data, list.result, linear.output, exclude)
    return(nn)
  }
#
varify.variables <-
  function(data, formula, startweights, learningrate.limit, learningrate.factor, 
            learningrate.bp, lifesign, algorithm, threshold, lifesign.step, 
            hidden, rep, stepmax, err.fct, act.fct) 
  {
    if (is.null(data)) 
      stop("'data' is missing", call. = FALSE)
    if (is.null(formula)) 
      stop("'formula' is missing", call. = FALSE)
    if (!is.null(startweights)) {
      startweights <- as.vector(unlist(startweights))
      if (any(is.na(startweights))) 
        startweights <- startweights[!is.na(startweights)]
    }
    data <- as.data.frame(data)
    formula <- stats::as.formula(formula)
    model.vars <- attr(stats::terms(formula), "term.labels")
    formula.reverse <- formula
    formula.reverse[[3]] <- formula[[2]]
    model.resp <- attr(stats::terms(formula.reverse), "term.labels")
    model.list <- list(response = model.resp, variables = model.vars)
    if (!is.null(learningrate.limit)) {
      if (length(learningrate.limit) != 2) 
        stop("'learningrate.factor' must consist of two components", 
             call. = FALSE)
      learningrate.limit <- as.list(learningrate.limit)
      names(learningrate.limit) <- c("min", "max")
      learningrate.limit$min <- as.vector(as.numeric(learningrate.limit$min))
      learningrate.limit$max <- as.vector(as.numeric(learningrate.limit$max))
      if (is.na(learningrate.limit$min) || is.na(learningrate.limit$max)) 
        stop("'learningrate.limit' must be a numeric vector", 
             call. = FALSE)
    }
    if (!is.null(learningrate.factor)) {
      if (length(learningrate.factor) != 2) 
        stop("'learningrate.factor' must consist of two components", 
             call. = FALSE)
      learningrate.factor <- as.list(learningrate.factor)
      names(learningrate.factor) <- c("minus", "plus")
      learningrate.factor$minus <- as.vector(as.numeric(learningrate.factor$minus))
      learningrate.factor$plus <- as.vector(as.numeric(learningrate.factor$plus))
      if (is.na(learningrate.factor$minus) || is.na(learningrate.factor$plus)) 
        stop("'learningrate.factor' must be a numeric vector", 
             call. = FALSE)
    }
    else learningrate.factor <- list(minus = c(0.5), plus = c(1.2))
    if (is.null(lifesign)) 
      lifesign <- "none"
    lifesign <- as.character(lifesign)
    if (!((lifesign == "none") || (lifesign == "minimal") || 
          (lifesign == "full"))) 
      lifesign <- "minimal"
    if (is.na(lifesign)) 
      stop("'lifesign' must be a character", call. = FALSE)
    if (is.null(algorithm)) 
      algorithm <- "rprop+"
    algorithm <- as.character(algorithm)
    if (!((algorithm == "rprop+") || (algorithm == "rprop-") || 
          (algorithm == "slr") || (algorithm == "sag") || (algorithm == 
                                                           "backprop"))) 
      stop("'algorithm' is not known", call. = FALSE)
    if (is.null(threshold)) 
      threshold <- 0.01
    threshold <- as.numeric(threshold)
    if (is.na(threshold)) 
      stop("'threshold' must be a numeric value", call. = FALSE)
    if (algorithm == "backprop") 
      if (is.null(learningrate.bp) || !is.numeric(learningrate.bp)) 
        stop("'learningrate' must be a numeric value, if the backpropagation algorithm is used", 
             call. = FALSE)
    if (is.null(lifesign.step)) 
      lifesign.step <- 1000
    lifesign.step <- as.integer(lifesign.step)
    if (is.na(lifesign.step)) 
      stop("'lifesign.step' must be an integer", call. = FALSE)
    if (lifesign.step < 1) 
      lifesign.step <- as.integer(100)
    if (is.null(hidden)) 
      hidden <- 0
    hidden <- as.vector(as.integer(hidden))
    if (prod(!is.na(hidden)) == 0) 
      stop("'hidden' must be an integer vector or a single integer", 
           call. = FALSE)
    if (length(hidden) > 1 && prod(hidden) == 0) 
      stop("'hidden' contains at least one 0", call. = FALSE)
    if (is.null(rep)) 
      rep <- 1
    rep <- as.integer(rep)
    if (is.na(rep)) 
      stop("'rep' must be an integer", call. = FALSE)
    if (is.null(stepmax)) 
      stepmax <- 10000
    stepmax <- as.integer(stepmax)
    if (is.na(stepmax)) 
      stop("'stepmax' must be an integer", call. = FALSE)
    if (stepmax < 1) 
      stepmax <- as.integer(1000)
    if (is.null(hidden)) {
      if (is.null(learningrate.limit)) 
        learningrate.limit <- list(min = c(1e-08), max = c(50))
    }
    else {
      if (is.null(learningrate.limit)) 
        learningrate.limit <- list(min = c(1e-10), max = c(0.1))
    }
    if (!is.function(act.fct) && act.fct != "logistic" && act.fct != 
        "tanh") 
      stop("''act.fct' is not known", call. = FALSE)
    if (!is.function(err.fct) && err.fct != "sse" && err.fct != 
        "ce") 
      stop("'err.fct' is not known", call. = FALSE)
    return(list(data = data, formula = formula, startweights = startweights, 
                learningrate.limit = learningrate.limit, learningrate.factor = learningrate.factor, 
                learningrate.bp = learningrate.bp, lifesign = lifesign, 
                algorithm = algorithm, threshold = threshold, lifesign.step = lifesign.step, 
                hidden = hidden, rep = rep, stepmax = stepmax, model.list = model.list))
  }
#
generate.initial.variables <-
  function(data, model.list, hidden, act.fct, err.fct, algorithm, 
            linear.output, formula) 
  {
    formula.reverse <- formula
    formula.reverse[[2]] <- stats::as.formula(paste(model.list$response[[1]], 
                                                    "~", model.list$variables[[1]], sep = ""))[[2]]
    formula.reverse[[3]] <- formula[[2]]
    response <- as.matrix(stats::model.frame(formula.reverse, data))
    formula.reverse[[3]] <- formula[[3]]
    covariate <- as.matrix(stats::model.frame(formula.reverse, data))
    covariate[, 1] <- 1
    colnames(covariate)[1] <- "intercept"
    if (is.function(act.fct)) {
      act.deriv.fct <- differentiate(act.fct)
      attr(act.fct, "type") <- "function"
    }
    else {
      if (act.fct == "tanh") {
        act.fct <- function(x) {
          tanh(x)
        }
        attr(act.fct, "type") <- "tanh"
        act.deriv.fct <- function(x) {
          1 - x^2
        }
      }
      else if (act.fct == "logistic") {
        act.fct <- function(x) {
          1/(1 + exp(-x))
        }
        attr(act.fct, "type") <- "logistic"
        act.deriv.fct <- function(x) {
          x * (1 - x)
        }
      }
    }
    if (is.function(err.fct)) {
      err.deriv.fct <- differentiate(err.fct)
      attr(err.fct, "type") <- "function"
    }
    else {
      if (err.fct == "ce") {
        if (all(response == 0 | response == 1)) {
          err.fct <- function(x, y) {
            -(y * log(x) + (1 - y) * log(1 - x))
          }
          attr(err.fct, "type") <- "ce"
          err.deriv.fct <- function(x, y) {
            (1 - y)/(1 - x) - y/x
          }
        }
        else {
          err.fct <- function(x, y) {
            1/2 * (y - x)^2
          }
          attr(err.fct, "type") <- "sse"
          err.deriv.fct <- function(x, y) {
            x - y
          }
          warning("'err.fct' was automatically set to sum of squared error (sse), because the response is not binary", 
                  call. = F)
        }
      }
      else if (err.fct == "sse") {
        err.fct <- function(x, y) {
          1/2 * (y - x)^2
        }
        attr(err.fct, "type") <- "sse"
        err.deriv.fct <- function(x, y) {
          x - y
        }
      }
    }
    return(list(covariate = covariate, response = response, err.fct = err.fct, 
                err.deriv.fct = err.deriv.fct, act.fct = act.fct, act.deriv.fct = act.deriv.fct))
  }
#
differentiate <-
  function(orig.fct, hessian = FALSE) 
  {
    body.fct <- deparse(body(orig.fct))
    if (body.fct[1] == "{") 
      body.fct <- body.fct[2]
    text <- paste("y~", body.fct, sep = "")
    text2 <- paste(deparse(orig.fct)[1], "{}")
    temp <- stats::deriv(eval(parse(text = text)), "x", func = eval(parse(text = text2)), 
                         hessian = hessian)
    temp <- deparse(temp)
    derivative <- NULL
    if (!hessian) 
      for (i in 1:length(temp)) {
        if (!any(grep("value", temp[i]))) 
          derivative <- c(derivative, temp[i])
      }
    else for (i in 1:length(temp)) {
      if (!any(grep("value", temp[i]), grep("grad", temp[i]), 
               grep(", c", temp[i]))) 
        derivative <- c(derivative, temp[i])
    }
    number <- NULL
    for (i in 1:length(derivative)) {
      if (any(grep("<-", derivative[i]))) 
        number <- i
    }
    if (is.null(number)) {
      return(function(x) {
        matrix(0, nrow(x), ncol(x))
      })
    }
    else {
      derivative[number] <- unlist(strsplit(derivative[number], 
                                            "<-"))[2]
      derivative <- eval(parse(text = derivative))
    }
    if (length(formals(derivative)) == 1 && length(derivative(c(1, 
                                                                1))) == 1) 
      derivative <- eval(parse(text = paste("function(x){matrix(", 
                                            derivative(1), ", nrow(x), ncol(x))}")))
    if (length(formals(derivative)) == 2 && length(derivative(c(1, 
                                                                1), c(1, 1))) == 1) 
      derivative <- eval(parse(text = paste("function(x, y){matrix(", 
                                            derivative(1, 1), ", nrow(x), ncol(x))}")))
    return(derivative)
  }
#
display <-
  function(hidden, threshold, rep, i.rep, lifesign) 
  {
    text <- paste("    rep: %", nchar(rep) - nchar(i.rep), "s", 
                  sep = "")
    cat("hidden: ", paste(hidden, collapse = ", "), "    thresh: ", 
        threshold, sprintf(eval(expression(text)), ""), i.rep, 
        "/", rep, "    steps: ", sep = "")
    if (lifesign == "full") 
      lifesign <- sum(nchar(hidden)) + 2 * length(hidden) - 
      2 + max(nchar(threshold)) + 2 * nchar(rep) + 41
    return(lifesign)
  }
#
calculate.neuralnet <-
  function(data, model.list, hidden, stepmax, rep, threshold, 
            learningrate.limit, learningrate.factor, lifesign, covariate, 
            response, lifesign.step, startweights, algorithm, act.fct, 
            act.deriv.fct, err.fct, err.deriv.fct, linear.output, likelihood, 
            exclude, constant.weights, learningrate.bp) 
  {
    time.start.local <- Sys.time()
    result <- generate.startweights(model.list, hidden, startweights, 
                                    rep, exclude, constant.weights)
    weights <- result$weights
    exclude <- result$exclude
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    result <- rprop(weights = weights, threshold = threshold, 
                    response = response, covariate = covariate, learningrate.limit = learningrate.limit, 
                    learningrate.factor = learningrate.factor, stepmax = stepmax, 
                    lifesign = lifesign, lifesign.step = lifesign.step, act.fct = act.fct, 
                    act.deriv.fct = act.deriv.fct, err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                    algorithm = algorithm, linear.output = linear.output, 
                    exclude = exclude, learningrate.bp = learningrate.bp)
    startweights <- weights
    weights <- result$weights
    step <- result$step
    reached.threshold <- result$reached.threshold
    net.result <- result$net.result
    error <- sum(err.fct(net.result, response))
    if (is.na(error) & type(err.fct) == "ce") 
      if (all(net.result <= 1, net.result >= 0)) 
        error <- sum(err.fct(net.result, response), na.rm = T)
    if (!is.null(constant.weights) && any(constant.weights != 
                                          0)) 
      exclude <- exclude[-which(constant.weights != 0)]
    if (length(exclude) == 0) 
      exclude <- NULL
    aic <- NULL
    bic <- NULL
    if (likelihood) {
      synapse.count <- length(unlist(weights)) - length(exclude)
      aic <- 2 * error + (2 * synapse.count)
      bic <- 2 * error + log(nrow(response)) * synapse.count
    }
    if (is.na(error)) 
      warning("'err.fct' does not fit 'data' or 'act.fct'", 
              call. = F)
    if (lifesign != "none") {
      if (reached.threshold <= threshold) {
        cat(rep(" ", (max(nchar(stepmax), nchar("stepmax")) - 
                        nchar(step))), step, sep = "")
        cat("\terror: ", round(error, 5), rep(" ", 6 - (nchar(round(error, 
                                                                    5)) - nchar(round(error, 0)))), sep = "")
        if (!is.null(aic)) {
          cat("\taic: ", round(aic, 5), rep(" ", 6 - (nchar(round(aic, 
                                                                  5)) - nchar(round(aic, 0)))), sep = "")
        }
        if (!is.null(bic)) {
          cat("\tbic: ", round(bic, 5), rep(" ", 6 - (nchar(round(bic, 
                                                                  5)) - nchar(round(bic, 0)))), sep = "")
        }
        time <- difftime(Sys.time(), time.start.local)
        cat("\ttime: ", round(time, 2), " ", attr(time, "units"), 
            sep = "")
        cat("\n")
      }
    }
    if (reached.threshold > threshold) 
      return(result = list(output.vector = NULL, weights = NULL))
    output.vector <- c(error = error, reached.threshold = reached.threshold, 
                       steps = step)
    if (!is.null(aic)) {
      output.vector <- c(output.vector, aic = aic)
    }
    if (!is.null(bic)) {
      output.vector <- c(output.vector, bic = bic)
    }
    for (w in 1:length(weights)) output.vector <- c(output.vector, 
                                                    as.vector(weights[[w]]))
    generalized.weights <- calculate.generalized.weights(weights, 
                                                         neuron.deriv = result$neuron.deriv, net.result = net.result)
    startweights <- unlist(startweights)
    weights <- unlist(weights)
    if (!is.null(exclude)) {
      startweights[exclude] <- NA
      weights[exclude] <- NA
    }
    startweights <- relist(startweights, nrow.weights, ncol.weights)
    weights <- relist(weights, nrow.weights, ncol.weights)
    return(list(generalized.weights = generalized.weights, weights = weights, 
                startweights = startweights, net.result = result$net.result, 
                output.vector = output.vector))
  }
#
generate.startweights <-
  function (model.list, hidden, startweights, rep, exclude, constant.weights) 
  {
    input.count <- length(model.list$variables)
    output.count <- length(model.list$response)
    if (!(length(hidden) == 1 && hidden == 0)) {
      length.weights <- length(hidden) + 1
      nrow.weights <- array(0, dim = c(length.weights))
      ncol.weights <- array(0, dim = c(length.weights))
      nrow.weights[1] <- (input.count + 1)
      ncol.weights[1] <- hidden[1]
      if (length(hidden) > 1) 
        for (i in 2:length(hidden)) {
          nrow.weights[i] <- hidden[i - 1] + 1
          ncol.weights[i] <- hidden[i]
        }
      nrow.weights[length.weights] <- hidden[length.weights - 
                                               1] + 1
      ncol.weights[length.weights] <- output.count
    }
    else {
      length.weights <- 1
      nrow.weights <- array((input.count + 1), dim = c(1))
      ncol.weights <- array(output.count, dim = c(1))
    }
    length <- sum(ncol.weights * nrow.weights)
    vector <- rep(0, length)
    if (!is.null(exclude)) {
      if (is.matrix(exclude)) {
        exclude <- matrix(as.integer(exclude), ncol = ncol(exclude), 
                          nrow = nrow(exclude))
        if (nrow(exclude) >= length || ncol(exclude) != 3) 
          stop("'exclude' has wrong dimensions", call. = FALSE)
        if (any(exclude < 1)) 
          stop("'exclude' contains at least one invalid weight", 
               call. = FALSE)
        temp <- relist(vector, nrow.weights, ncol.weights)
        for (i in 1:nrow(exclude)) {
          if (exclude[i, 1] > length.weights || exclude[i, 
                                                        2] > nrow.weights[exclude[i, 1]] || exclude[i, 
                                                                                                    3] > ncol.weights[exclude[i, 1]]) 
            stop("'exclude' contains at least one invalid weight", 
                 call. = FALSE)
          temp[[exclude[i, 1]]][exclude[i, 2], exclude[i, 
                                                       3]] <- 1
        }
        exclude <- which(unlist(temp) == 1)
      }
      else if (is.vector(exclude)) {
        exclude <- as.integer(exclude)
        if (max(exclude) > length || min(exclude) < 1) {
          stop("'exclude' contains at least one invalid weight", 
               call. = FALSE)
        }
      }
      else {
        stop("'exclude' must be a vector or matrix", call. = FALSE)
      }
      if (length(exclude) >= length) 
        stop("all weights are exluded", call. = FALSE)
    }
    length <- length - length(exclude)
    if (!is.null(exclude)) {
      if (is.null(startweights) || length(startweights) < (length * 
                                                           rep)) 
        vector[-exclude] <- stats::rnorm(length)
      else vector[-exclude] <- startweights[((rep - 1) * length + 
                                               1):(length * rep)]
    }
    else {
      if (is.null(startweights) || length(startweights) < (length * 
                                                           rep)) 
        vector <- stats::rnorm(length)
      else vector <- startweights[((rep - 1) * length + 1):(length * 
                                                              rep)]
    }
    if (!is.null(exclude) && !is.null(constant.weights)) {
      if (length(exclude) < length(constant.weights)) 
        stop("constant.weights contains more weights than exclude", 
             call. = FALSE)
      else vector[exclude[1:length(constant.weights)]] <- constant.weights
    }
    weights <- relist(vector, nrow.weights, ncol.weights)
    return(list(weights = weights, exclude = exclude))
  }
#
rprop <-
  function (weights, response, covariate, threshold, learningrate.limit, 
            learningrate.factor, stepmax, lifesign, lifesign.step, act.fct, 
            act.deriv.fct, err.fct, err.deriv.fct, algorithm, linear.output, 
            exclude, learningrate.bp) 
  {
    step <- 1
    nchar.stepmax <- max(nchar(stepmax), 7)
    length.weights <- length(weights)
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    length.unlist <- length(unlist(weights)) - length(exclude)
    learningrate <- as.vector(matrix(0.1, nrow = 1, ncol = length.unlist))
    gradients.old <- as.vector(matrix(0, nrow = 1, ncol = length.unlist))
    if (is.null(exclude)) 
      exclude <- length(unlist(weights)) + 1
    if (type(act.fct) == "tanh" || type(act.fct) == "logistic") 
      special <- TRUE
    else special <- FALSE
    if (linear.output) {
      output.act.fct <- function(x) {
        x
      }
      output.act.deriv.fct <- function(x) {
        matrix(1, nrow(x), ncol(x))
      }
    }
    else {
      if (type(err.fct) == "ce" && type(act.fct) == "logistic") {
        err.deriv.fct <- function(x, y) {
          x * (1 - y) - y * (1 - x)
        }
        linear.output <- TRUE
      }
      output.act.fct <- act.fct
      output.act.deriv.fct <- act.deriv.fct
    }
    result <- compute.net(weights, length.weights, covariate = covariate, 
                          act.fct = act.fct, act.deriv.fct = act.deriv.fct, output.act.fct = output.act.fct, 
                          output.act.deriv.fct = output.act.deriv.fct, special)
    err.deriv <- err.deriv.fct(result$net.result, response)
    gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
                                     neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
                                     err.deriv = err.deriv, exclude = exclude, linear.output = linear.output)
    reached.threshold <- max(abs(gradients))
    min.reached.threshold <- reached.threshold
    while (step < stepmax && reached.threshold > threshold) {
      if (!is.character(lifesign) && step %% lifesign.step == 
          0) {
        text <- paste("%", nchar.stepmax, "s", sep = "")
        cat(sprintf(eval(expression(text)), step), "\tmin thresh: ", 
            min.reached.threshold, "\n", rep(" ", lifesign), 
            sep = "")
        utils::flush.console()
      }
      if (algorithm == "rprop+") 
        result <- plus(gradients, gradients.old, weights, 
                       nrow.weights, ncol.weights, learningrate, learningrate.factor, 
                       learningrate.limit, exclude)
      else if (algorithm == "backprop") 
        result <- backprop(gradients, weights, length.weights, 
                           nrow.weights, ncol.weights, learningrate.bp, 
                           exclude)
      else result <- minus(gradients, gradients.old, weights, 
                           length.weights, nrow.weights, ncol.weights, learningrate, 
                           learningrate.factor, learningrate.limit, algorithm, 
                           exclude)
      gradients.old <- result$gradients.old
      weights <- result$weights
      learningrate <- result$learningrate
      result <- compute.net(weights, length.weights, covariate = covariate, 
                            act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                            output.act.fct = output.act.fct, output.act.deriv.fct = output.act.deriv.fct, 
                            special)
      err.deriv <- err.deriv.fct(result$net.result, response)
      gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
                                       neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
                                       err.deriv = err.deriv, exclude = exclude, linear.output = linear.output)
      reached.threshold <- max(abs(gradients))
      if (reached.threshold < min.reached.threshold) {
        min.reached.threshold <- reached.threshold
      }
      step <- step + 1
    }
    if (lifesign != "none" && reached.threshold > threshold) {
      cat("stepmax\tmin thresh: ", min.reached.threshold, "\n", 
          sep = "")
    }
    return(list(weights = weights, step = as.integer(step), reached.threshold = reached.threshold, 
                net.result = result$net.result, neuron.deriv = result$neuron.deriv))
  }
#
compute.net <-
  function(weights, length.weights, covariate, act.fct, act.deriv.fct, 
            output.act.fct, output.act.deriv.fct, special) 
  {
    neuron.deriv <- NULL
    neurons <- list(covariate)
    if (length.weights > 1) 
      for (i in 1:(length.weights - 1)) {
        temp <- neurons[[i]] %*% weights[[i]]
        act.temp <- act.fct(temp)
        if (special) 
          neuron.deriv[[i]] <- act.deriv.fct(act.temp)
        else neuron.deriv[[i]] <- act.deriv.fct(temp)
        neurons[[i + 1]] <- cbind(1, act.temp)
      }
    if (!is.list(neuron.deriv)) 
      neuron.deriv <- list(neuron.deriv)
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    net.result <- output.act.fct(temp)
    if (special) 
      neuron.deriv[[length.weights]] <- output.act.deriv.fct(net.result)
    else neuron.deriv[[length.weights]] <- output.act.deriv.fct(temp)
    if (any(is.na(neuron.deriv))) 
      stop("neuron derivatives contain a NA; varify that the derivative function does not divide by 0", 
           call. = FALSE)
    list(neurons = neurons, neuron.deriv = neuron.deriv, net.result = net.result)
  }
#
calculate.gradients <-
  function(weights, length.weights, neurons, neuron.deriv, err.deriv, 
            exclude, linear.output) 
  {
    if (any(is.na(err.deriv))) 
      stop("the error derivative contains a NA; varify that the derivative function does not divide by 0 (e.g. cross entropy)", 
           call. = FALSE)
    if (!linear.output) 
      delta <- neuron.deriv[[length.weights]] * err.deriv
    else delta <- err.deriv
    gradients <- crossprod(neurons[[length.weights]], delta)
    if (length.weights > 1) 
      for (w in (length.weights - 1):1) {
        delta <- neuron.deriv[[w]] * tcrossprod(delta, remove.intercept(weights[[w + 
                                                                                   1]]))
        gradients <- c(crossprod(neurons[[w]], delta), gradients)
      }
    gradients[-exclude]
  }
#
plus <-
  function(gradients, gradients.old, weights, nrow.weights, ncol.weights, 
            learningrate, learningrate.factor, learningrate.limit, exclude) 
  {
    weights <- unlist(weights)
    sign.gradient <- sign(gradients)
    temp <- gradients.old * sign.gradient
    positive <- temp > 0
    negative <- temp < 0
    not.negative <- !negative
    if (any(positive)) {
      learningrate[positive] <- pmin.int(learningrate[positive] * 
                                           learningrate.factor$plus, learningrate.limit$max)
    }
    if (any(negative)) {
      weights[-exclude][negative] <- weights[-exclude][negative] + 
        gradients.old[negative] * learningrate[negative]
      learningrate[negative] <- pmax.int(learningrate[negative] * 
                                           learningrate.factor$minus, learningrate.limit$min)
      gradients.old[negative] <- 0
      if (any(not.negative)) {
        weights[-exclude][not.negative] <- weights[-exclude][not.negative] - 
          sign.gradient[not.negative] * learningrate[not.negative]
        gradients.old[not.negative] <- sign.gradient[not.negative]
      }
    }
    else {
      weights[-exclude] <- weights[-exclude] - sign.gradient * 
        learningrate
      gradients.old <- sign.gradient
    }
    list(gradients.old = gradients.old, weights = relist(weights, 
                                                         nrow.weights, ncol.weights), learningrate = learningrate)
  }
#
backprop <-
  function(gradients, weights, length.weights, nrow.weights, ncol.weights, 
            learningrate.bp, exclude) 
  {
    weights <- unlist(weights)
    if (!is.null(exclude)) 
      weights[-exclude] <- weights[-exclude] - gradients * 
        learningrate.bp
    else weights <- weights - gradients * learningrate.bp
    list(gradients.old = gradients, weights = relist(weights, 
                                                     nrow.weights, ncol.weights), learningrate = learningrate.bp)
  }
#
minus <-
  function(gradients, gradients.old, weights, length.weights, 
            nrow.weights, ncol.weights, learningrate, learningrate.factor, 
            learningrate.limit, algorithm, exclude) 
  {
    weights <- unlist(weights)
    temp <- gradients.old * gradients
    positive <- temp > 0
    negative <- temp < 0
    if (any(positive)) 
      learningrate[positive] <- pmin.int(learningrate[positive] * 
                                           learningrate.factor$plus, learningrate.limit$max)
    if (any(negative)) 
      learningrate[negative] <- pmax.int(learningrate[negative] * 
                                           learningrate.factor$minus, learningrate.limit$min)
    if (algorithm != "rprop-") {
      delta <- 10^-6
      notzero <- gradients != 0
      gradients.notzero <- gradients[notzero]
      if (algorithm == "slr") {
        min <- which.min(learningrate[notzero])
      }
      else if (algorithm == "sag") {
        min <- which.min(abs(gradients.notzero))
      }
      if (length(min) != 0) {
        temp <- learningrate[notzero] * gradients.notzero
        sum <- sum(temp[-min]) + delta
        learningrate[notzero][min] <- min(max(-sum/gradients.notzero[min], 
                                              learningrate.limit$min), learningrate.limit$max)
      }
    }
    weights[-exclude] <- weights[-exclude] - sign(gradients) * 
      learningrate
    list(gradients.old = gradients, weights = relist(weights, 
                                                     nrow.weights, ncol.weights), learningrate = learningrate)
  }
#
calculate.generalized.weights <-
  function(weights, neuron.deriv, net.result) 
  {
    for (w in 1:length(weights)) {
      weights[[w]] <- remove.intercept(weights[[w]])
    }
    generalized.weights <- NULL
    for (k in 1:ncol(net.result)) {
      for (w in length(weights):1) {
        if (w == length(weights)) {
          temp <- neuron.deriv[[length(weights)]][, k] * 
            1/(net.result[, k] * (1 - (net.result[, k])))
          delta <- tcrossprod(temp, weights[[w]][, k])
        }
        else {
          delta <- tcrossprod(delta * neuron.deriv[[w]], 
                              weights[[w]])
        }
      }
      generalized.weights <- cbind(generalized.weights, delta)
    }
    return(generalized.weights)
  }
#
generate.output <-
  function(covariate, call, rep, threshold, matrix, startweights, 
            model.list, response, err.fct, act.fct, data, list.result, 
            linear.output, exclude) 
  {
    covariate <- t(remove.intercept(t(covariate)))
    nn <- list(call = call)
    class(nn) <- c("nn")
    nn$response <- response
    nn$covariate <- covariate
    nn$model.list <- model.list
    nn$err.fct <- err.fct
    nn$act.fct <- act.fct
    nn$linear.output <- linear.output
    nn$data <- data
    nn$exclude <- exclude
    if (!is.null(matrix)) {
      nn$net.result <- NULL
      nn$weights <- NULL
      nn$generalized.weights <- NULL
      nn$startweights <- NULL
      for (i in 1:length(list.result)) {
        nn$net.result <- c(nn$net.result, list(list.result[[i]]$net.result))
        nn$weights <- c(nn$weights, list(list.result[[i]]$weights))
        nn$startweights <- c(nn$startweights, list(list.result[[i]]$startweights))
        nn$generalized.weights <- c(nn$generalized.weights, 
                                    list(list.result[[i]]$generalized.weights))
      }
      nn$result.matrix <- generate.rownames(matrix, nn$weights[[1]], 
                                            model.list)
    }
    return(nn)
  }
#
generate.rownames <-
  function(matrix, weights, model.list) 
  {
    rownames <- rownames(matrix)[rownames(matrix) != ""]
    for (w in 1:length(weights)) {
      for (j in 1:ncol(weights[[w]])) {
        for (i in 1:nrow(weights[[w]])) {
          if (i == 1) {
            if (w == length(weights)) {
              rownames <- c(rownames, paste("Intercept.to.", 
                                            model.list$response[j], sep = ""))
            }
            else {
              rownames <- c(rownames, paste("Intercept.to.", 
                                            w, "layhid", j, sep = ""))
            }
          }
          else {
            if (w == 1) {
              if (w == length(weights)) {
                rownames <- c(rownames, paste(model.list$variables[i - 
                                                                     1], ".to.", model.list$response[j], sep = ""))
              }
              else {
                rownames <- c(rownames, paste(model.list$variables[i - 
                                                                     1], ".to.1layhid", j, sep = ""))
              }
            }
            else {
              if (w == length(weights)) {
                rownames <- c(rownames, paste(w - 1, "layhid.", 
                                              i - 1, ".to.", model.list$response[j], 
                                              sep = ""))
              }
              else {
                rownames <- c(rownames, paste(w - 1, "layhid.", 
                                              i - 1, ".to.", w, "layhid", j, sep = ""))
              }
            }
          }
        }
      }
    }
    rownames(matrix) <- rownames
    colnames(matrix) <- 1:(ncol(matrix))
    return(matrix)
  }
#
relist <-
  function(x, nrow, ncol) 
  {
    list.x <- NULL
    for (w in 1:length(nrow)) {
      length <- nrow[w] * ncol[w]
      list.x[[w]] <- matrix(x[1:length], nrow = nrow[w], ncol = ncol[w])
      x <- x[-(1:length)]
    }
    list.x
  }
#
remove.intercept <-
  function (matrix) 
  {
    matrix(matrix[-1, ], ncol = ncol(matrix))
  }
#
type <-
  function (fct) 
  {
    attr(fct, "type")
  }
#
print.nn <-
  function(x, ...) 
  {
    matrix <- x$result.matrix
    cat("Call: ", deparse(x$call), "\n\n", sep = "")
    if (!is.null(matrix)) {
      if (ncol(matrix) > 1) {
        cat(ncol(matrix), " repetitions were calculated.\n\n", 
            sep = "")
        sorted.matrix <- matrix[, order(matrix["error", ])]
        if (any(rownames(sorted.matrix) == "aic")) {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], AIC = sorted.matrix["aic", ], BIC = sorted.matrix["bic", 
                                                                                                   ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                                                                          ], Steps = sorted.matrix["steps", ])))
        }
        else {
          print(t(rbind(Error = sorted.matrix["error", 
                                              ], `Reached Threshold` = sorted.matrix["reached.threshold", 
                                                                                     ], Steps = sorted.matrix["steps", ])))
        }
      }
      else {
        cat(ncol(matrix), " repetition was calculated.\n\n", 
            sep = "")
        if (any(rownames(matrix) == "aic")) {
          print(t(matrix(c(matrix["error", ], matrix["aic", 
                                                     ], matrix["bic", ], matrix["reached.threshold", 
                                                                                ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                                                         "AIC", "BIC", "Reached Threshold", "Steps"), 
                                                                                                                       c(1)))))
        }
        else {
          print(t(matrix(c(matrix["error", ], matrix["reached.threshold", 
                                                     ], matrix["steps", ]), dimnames = list(c("Error", 
                                                                                              "Reached Threshold", "Steps"), c(1)))))
        }
      }
    }
    cat("\n")
  }
#
compute <-
  function(x, covariate, rep = 1) 
  {
    nn <- x
    linear.output <- nn$linear.output
    weights <- nn$weights[[rep]]
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    weights <- unlist(weights)
    if (any(is.na(weights))) 
      weights[is.na(weights)] <- 0
    weights <- relist(weights, nrow.weights, ncol.weights)
    length.weights <- length(weights)
    covariate <- as.matrix(cbind(1, covariate))
    act.fct <- nn$act.fct
    neurons <- list(covariate)
    if (length.weights > 1) 
      for (i in 1:(length.weights - 1)) {
        temp <- neurons[[i]] %*% weights[[i]]
        act.temp <- act.fct(temp)
        neurons[[i + 1]] <- cbind(1, act.temp)
      }
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    if (linear.output) 
      net.result <- temp
    else net.result <- act.fct(temp)
    list(neurons = neurons, net.result = net.result)
  }
#
prediction <-
  function(x, list.glm = NULL) 
  {
    nn <- x
    data.result <- calculate.data.result(response = nn$response, 
                                         model.list = nn$model.list, covariate = nn$covariate)
    predictions <- calculate.predictions(covariate = nn$covariate, 
                                         data.result = data.result, list.glm = list.glm, matrix = nn$result.matrix, 
                                         list.net.result = nn$net.result, model.list = nn$model.list)
    if (type(nn$err.fct) == "ce" && all(data.result >= 0) && 
        all(data.result <= 1)) 
      data.error <- sum(nn$err.fct(data.result, nn$response), 
                        na.rm = T)
    else data.error <- sum(nn$err.fct(data.result, nn$response))
    cat("Data Error:\t", data.error, ";\n", sep = "")
    predictions
  }
#
calculate.predictions <-
  function(covariate, data.result, list.glm, matrix, list.net.result, 
            model.list) 
  {
    not.duplicated <- !duplicated(covariate)
    nrow.notdupl <- sum(not.duplicated)
    covariate.mod <- matrix(covariate[not.duplicated, ], nrow = nrow.notdupl)
    predictions <- list(data = cbind(covariate.mod, matrix(data.result[not.duplicated, 
                                                                       ], nrow = nrow.notdupl)))
    if (!is.null(matrix)) {
      for (i in length(list.net.result):1) {
        pred.temp <- cbind(covariate.mod, matrix(list.net.result[[i]][not.duplicated, 
                                                                      ], nrow = nrow.notdupl))
        predictions <- eval(parse(text = paste("c(list(rep", 
                                               i, "=pred.temp), predictions)", sep = "")))
      }
    }
    if (!is.null(list.glm)) {
      for (i in 1:length(list.glm)) {
        pred.temp <- cbind(covariate.mod, matrix(list.glm[[i]]$fitted.values[not.duplicated], 
                                                 nrow = nrow.notdupl))
        text <- paste("c(predictions, list(glm.", names(list.glm[i]), 
                      "=pred.temp))", sep = "")
        predictions <- eval(parse(text = text))
      }
    }
    for (i in 1:length(predictions)) {
      colnames(predictions[[i]]) <- c(model.list$variables, 
                                      model.list$response)
      if (nrow(covariate) > 1) 
        for (j in (1:ncol(covariate))) predictions[[i]] <- predictions[[i]][order(predictions[[i]][, 
                                                                                                   j]), ]
        rownames(predictions[[i]]) <- 1:nrow(predictions[[i]])
    }
    predictions
  }
#
calculate.data.result <-
  function(response, covariate, model.list) 
  {
    duplicated <- duplicated(covariate)
    if (!any(duplicated)) {
      return(response)
    }
    which.duplicated <- seq_along(duplicated)[duplicated]
    which.not.duplicated <- seq_along(duplicated)[!duplicated]
    ncol.response <- ncol(response)
    if (ncol(covariate) == 1) {
      for (each in which.not.duplicated) {
        out <- NULL
        if (length(which.duplicated) > 0) {
          out <- covariate[which.duplicated, ] == covariate[each, 
                                                            ]
          if (any(out)) {
            rows <- c(each, which.duplicated[out])
            response[rows, ] = matrix(colMeans(matrix(response[rows, 
                                                               ], ncol = ncol.response)), ncol = ncol.response, 
                                      nrow = length(rows), byrow = T)
            which.duplicated <- which.duplicated[-out]
          }
        }
      }
    }
    else {
      tcovariate <- t(covariate)
      for (each in which.not.duplicated) {
        out <- NULL
        if (length(which.duplicated) > 0) {
          out <- apply(tcovariate[, which.duplicated] == 
                         covariate[each, ], 2, FUN = all)
          if (any(out)) {
            rows <- c(each, which.duplicated[out])
            response[rows, ] = matrix(colMeans(matrix(response[rows, 
                                                               ], ncol = ncol.response)), ncol = ncol.response, 
                                      nrow = length(rows), byrow = T)
            which.duplicated <- which.duplicated[-out]
          }
        }
      }
    }
    response
  }
#
plot.nn <-
  function(x, rep = NULL, x.entry = NULL, x.out = NULL, radius = 0.15, 
            arrow.length = 0.2, intercept = TRUE, intercept.factor = 0.4, 
            information = TRUE, information.pos = 0.1, col.entry.synapse = "black", 
            col.entry = "black", col.hidden = "black", col.hidden.synapse = "black", 
            col.out = "black", col.out.synapse = "black", col.intercept = "blue", 
            fontsize = 12, dimension = 6, show.weights = TRUE, file = NULL, 
            ...) 
  {
    net <- x
    if (is.null(net$weights)) 
      stop("weights were not calculated")
    if (!is.null(file) && !is.character(file)) 
      stop("'file' must be a string")
    if (is.null(rep)) {
      for (i in 1:length(net$weights)) {
        if (!is.null(file)) 
          file.rep <- paste(file, ".", i, sep = "")
        else file.rep <- NULL
        grDevices::dev.new()
        plot.nn(net, rep = i, x.entry, x.out, radius, arrow.length, 
                intercept, intercept.factor, information, information.pos, 
                col.entry.synapse, col.entry, col.hidden, col.hidden.synapse, 
                col.out, col.out.synapse, col.intercept, fontsize, 
                dimension, show.weights, file.rep, ...)
      }
    }
    else {
      if (is.character(file) && file.exists(file)) 
        stop(sprintf("%s already exists", sQuote(file)))
      result.matrix <- t(net$result.matrix)
      if (rep == "best") 
        rep <- as.integer(which.min(result.matrix[, "error"]))
      if (rep > length(net$weights)) 
        stop("'rep' does not exist")
      weights <- net$weights[[rep]]
      if (is.null(x.entry)) 
        x.entry <- 0.5 - (arrow.length/2) * length(weights)
      if (is.null(x.out)) 
        x.out <- 0.5 + (arrow.length/2) * length(weights)
      width <- max(x.out - x.entry + 0.2, 0.8) * 8
      radius <- radius/dimension
      entry.label <- net$model.list$variables
      out.label <- net$model.list$response
      neuron.count <- array(0, length(weights) + 1)
      neuron.count[1] <- nrow(weights[[1]]) - 1
      neuron.count[2] <- ncol(weights[[1]])
      x.position <- array(0, length(weights) + 1)
      x.position[1] <- x.entry
      x.position[length(weights) + 1] <- x.out
      if (length(weights) > 1) 
        for (i in 2:length(weights)) {
          neuron.count[i + 1] <- ncol(weights[[i]])
          x.position[i] <- x.entry + (i - 1) * (x.out - 
                                                  x.entry)/length(weights)
        }
      y.step <- 1/(neuron.count + 1)
      y.position <- array(0, length(weights) + 1)
      y.intercept <- 1 - 2 * radius
      information.pos <- min(min(y.step) - 0.1, 0.2)
      if (length(entry.label) != neuron.count[1]) {
        if (length(entry.label) < neuron.count[1]) {
          tmp <- NULL
          for (i in 1:(neuron.count[1] - length(entry.label))) {
            tmp <- c(tmp, "no name")
          }
          entry.label <- c(entry.label, tmp)
        }
      }
      if (length(out.label) != neuron.count[length(neuron.count)]) {
        if (length(out.label) < neuron.count[length(neuron.count)]) {
          tmp <- NULL
          for (i in 1:(neuron.count[length(neuron.count)] - 
                       length(out.label))) {
            tmp <- c(tmp, "no name")
          }
          out.label <- c(out.label, tmp)
        }
      }
      grid::grid.newpage()
      for (k in 1:length(weights)) {
        for (i in 1:neuron.count[k]) {
          y.position[k] <- y.position[k] + y.step[k]
          y.tmp <- 0
          for (j in 1:neuron.count[k + 1]) {
            y.tmp <- y.tmp + y.step[k + 1]
            result <- calculate.delta(c(x.position[k], 
                                        x.position[k + 1]), c(y.position[k], y.tmp), 
                                      radius)
            x <- c(x.position[k], x.position[k + 1] - result[1])
            y <- c(y.position[k], y.tmp + result[2])
            grid::grid.lines(x = x, y = y, arrow = grid::arrow(length = grid::unit(0.15, 
                                                                                   "cm"), type = "closed"), gp = grid::gpar(fill = col.hidden.synapse, 
                                                                                                                            col = col.hidden.synapse, ...))
            if (show.weights) 
              draw.text(label = weights[[k]][neuron.count[k] - 
                                               i + 2, neuron.count[k + 1] - j + 1], x = c(x.position[k], 
                                                                                          x.position[k + 1]), y = c(y.position[k], 
                                                                                                                    y.tmp), xy.null = 1.25 * result, color = col.hidden.synapse, 
                        fontsize = fontsize - 2, ...)
          }
          if (k == 1) {
            grid::grid.lines(x = c((x.position[1] - arrow.length), 
                                   x.position[1] - radius), y = y.position[k], 
                             arrow = grid::arrow(length = grid::unit(0.15, "cm"), 
                                                 type = "closed"), gp = grid::gpar(fill = col.entry.synapse, 
                                                                                   col = col.entry.synapse, ...))
            draw.text(label = entry.label[(neuron.count[1] + 
                                             1) - i], x = c((x.position - arrow.length), 
                                                            x.position[1] - radius), y = c(y.position[k], 
                                                                                           y.position[k]), xy.null = c(0, 0), color = col.entry.synapse, 
                      fontsize = fontsize, ...)
            grid::grid.circle(x = x.position[k], y = y.position[k], 
                              r = radius, gp = grid::gpar(fill = "white", col = col.entry, 
                                                          ...))
          }
          else {
            grid::grid.circle(x = x.position[k], y = y.position[k], 
                              r = radius, gp = grid::gpar(fill = "white", col = col.hidden, 
                                                          ...))
          }
        }
      }
      out <- length(neuron.count)
      for (i in 1:neuron.count[out]) {
        y.position[out] <- y.position[out] + y.step[out]
        grid::grid.lines(x = c(x.position[out] + radius, x.position[out] + 
                                 arrow.length), y = y.position[out], arrow = grid::arrow(length = grid::unit(0.15, 
                                                                                                             "cm"), type = "closed"), gp = grid::gpar(fill = col.out.synapse, 
                                                                                                                                                      col = col.out.synapse, ...))
        draw.text(label = out.label[(neuron.count[out] + 
                                       1) - i], x = c((x.position[out] + radius), x.position[out] + 
                                                        arrow.length), y = c(y.position[out], y.position[out]), 
                  xy.null = c(0, 0), color = col.out.synapse, fontsize = fontsize, 
                  ...)
        grid::grid.circle(x = x.position[out], y = y.position[out], 
                          r = radius, gp = grid::gpar(fill = "white", col = col.out, 
                                                      ...))
      }
      if (intercept) {
        for (k in 1:length(weights)) {
          y.tmp <- 0
          x.intercept <- (x.position[k + 1] - x.position[k]) * 
            intercept.factor + x.position[k]
          for (i in 1:neuron.count[k + 1]) {
            y.tmp <- y.tmp + y.step[k + 1]
            result <- calculate.delta(c(x.intercept, x.position[k + 
                                                                  1]), c(y.intercept, y.tmp), radius)
            x <- c(x.intercept, x.position[k + 1] - result[1])
            y <- c(y.intercept, y.tmp + result[2])
            grid::grid.lines(x = x, y = y, arrow = grid::arrow(length = grid::unit(0.15, 
                                                                                   "cm"), type = "closed"), gp = grid::gpar(fill = col.intercept, 
                                                                                                                            col = col.intercept, ...))
            xy.null <- cbind(x.position[k + 1] - x.intercept - 
                               2 * result[1], -(y.tmp - y.intercept + 2 * 
                                                  result[2]))
            if (show.weights) 
              draw.text(label = weights[[k]][1, neuron.count[k + 
                                                               1] - i + 1], x = c(x.intercept, x.position[k + 
                                                                                                            1]), y = c(y.intercept, y.tmp), xy.null = xy.null, 
                        color = col.intercept, alignment = c("right", 
                                                             "bottom"), fontsize = fontsize - 2, ...)
          }
          grid::grid.circle(x = x.intercept, y = y.intercept, 
                            r = radius, gp = grid::gpar(fill = "white", col = col.intercept, 
                                                        ...))
          grid::grid.text(1, x = x.intercept, y = y.intercept, 
                          gp = grid::gpar(col = col.intercept, ...))
        }
      }
      if (information) 
        grid::grid.text(paste("Error: ", round(result.matrix[rep, 
                                                             "error"], 6), "   Steps: ", result.matrix[rep, 
                                                                                                       "steps"], sep = ""), x = 0.5, y = information.pos, 
                        just = "bottom", gp = grid::gpar(fontsize = fontsize + 
                                                           2, ...))
      if (!is.null(file)) {
        weight.plot <- grDevices::recordPlot()
        save(weight.plot, file = file)
      }
    }
  }
#
calculate.delta <-
  function(x, y, r) 
  {
    delta.x <- x[2] - x[1]
    delta.y <- y[2] - y[1]
    x.null <- r/sqrt(delta.x^2 + delta.y^2) * delta.x
    if (y[1] < y[2]) 
      y.null <- -sqrt(r^2 - x.null^2)
    else if (y[1] > y[2]) 
      y.null <- sqrt(r^2 - x.null^2)
    else y.null <- 0
    c(x.null, y.null)
  }
#
draw.text <-
  function(label, x, y, xy.null = c(0, 0), color, alignment = c("left", 
                                                                 "bottom"), ...) 
  {
    x.label <- x[1] + xy.null[1]
    y.label <- y[1] - xy.null[2]
    x.delta <- x[2] - x[1]
    y.delta <- y[2] - y[1]
    angle = atan(y.delta/x.delta) * (180/pi)
    if (angle < 0) 
      angle <- angle + 0
    else if (angle > 0) 
      angle <- angle - 0
    if (is.numeric(label)) 
      label <- round(label, 5)
    vp <- grid::viewport(x = x.label, y = y.label, width = 0, height = , 
                         angle = angle, name = "vp1", just = alignment)
    grid::grid.text(label, x = 0, y = grid::unit(0.75, "mm"), just = alignment, 
                    gp = grid::gpar(col = color, ...), vp = vp)
  }
#
prediction <-
  function(x, list.glm = NULL) 
  {
    nn <- x
    data.result <- calculate.data.result(response = nn$response, 
                                         model.list = nn$model.list, covariate = nn$covariate)
    predictions <- calculate.predictions(covariate = nn$covariate, 
                                         data.result = data.result, list.glm = list.glm, matrix = nn$result.matrix, 
                                         list.net.result = nn$net.result, model.list = nn$model.list)
    if (type(nn$err.fct) == "ce" && all(data.result >= 0) && 
        all(data.result <= 1)) 
      data.error <- sum(nn$err.fct(data.result, nn$response), 
                        na.rm = T)
    else data.error <- sum(nn$err.fct(data.result, nn$response))
    cat("Data Error:\t", data.error, ";\n", sep = "")
    predictions
  }
#
calculate.predictions <-
  function(covariate, data.result, list.glm, matrix, list.net.result, 
            model.list) 
  {
    not.duplicated <- !duplicated(covariate)
    nrow.notdupl <- sum(not.duplicated)
    covariate.mod <- matrix(covariate[not.duplicated, ], nrow = nrow.notdupl)
    predictions <- list(data = cbind(covariate.mod, matrix(data.result[not.duplicated, 
                                                                       ], nrow = nrow.notdupl)))
    if (!is.null(matrix)) {
      for (i in length(list.net.result):1) {
        pred.temp <- cbind(covariate.mod, matrix(list.net.result[[i]][not.duplicated, 
                                                                      ], nrow = nrow.notdupl))
        predictions <- eval(parse(text = paste("c(list(rep", 
                                               i, "=pred.temp), predictions)", sep = "")))
      }
    }
    if (!is.null(list.glm)) {
      for (i in 1:length(list.glm)) {
        pred.temp <- cbind(covariate.mod, matrix(list.glm[[i]]$fitted.values[not.duplicated], 
                                                 nrow = nrow.notdupl))
        text <- paste("c(predictions, list(glm.", names(list.glm[i]), 
                      "=pred.temp))", sep = "")
        predictions <- eval(parse(text = text))
      }
    }
    for (i in 1:length(predictions)) {
      colnames(predictions[[i]]) <- c(model.list$variables, 
                                      model.list$response)
      if (nrow(covariate) > 1) 
        for (j in (1:ncol(covariate))) predictions[[i]] <- predictions[[i]][order(predictions[[i]][, 
                                                                                                   j]), ]
        rownames(predictions[[i]]) <- 1:nrow(predictions[[i]])
    }
    predictions
  }
#
calculate.data.result <-
  function(response, covariate, model.list) 
  {
    duplicated <- duplicated(covariate)
    if (!any(duplicated)) {
      return(response)
    }
    which.duplicated <- seq_along(duplicated)[duplicated]
    which.not.duplicated <- seq_along(duplicated)[!duplicated]
    ncol.response <- ncol(response)
    if (ncol(covariate) == 1) {
      for (each in which.not.duplicated) {
        out <- NULL
        if (length(which.duplicated) > 0) {
          out <- covariate[which.duplicated, ] == covariate[each, 
                                                            ]
          if (any(out)) {
            rows <- c(each, which.duplicated[out])
            response[rows, ] = matrix(colMeans(matrix(response[rows, 
                                                               ], ncol = ncol.response)), ncol = ncol.response, 
                                      nrow = length(rows), byrow = T)
            which.duplicated <- which.duplicated[-out]
          }
        }
      }
    }
    else {
      tcovariate <- t(covariate)
      for (each in which.not.duplicated) {
        out <- NULL
        if (length(which.duplicated) > 0) {
          out <- apply(tcovariate[, which.duplicated] == 
                         covariate[each, ], 2, FUN = all)
          if (any(out)) {
            rows <- c(each, which.duplicated[out])
            response[rows, ] = matrix(colMeans(matrix(response[rows, 
                                                               ], ncol = ncol.response)), ncol = ncol.response, 
                                      nrow = length(rows), byrow = T)
            which.duplicated <- which.duplicated[-out]
          }
        }
      }
    }
    response
  }
#
# get data
#
  setwd("C:/EAF LLC/aa-Analytics and BI/Machine Learning/Azure ML Studio")
#
  Sales <- read.csv("20171018 training data rev2.csv",
                    header = TRUE)
  Future_Sales <- read.csv("20171018 test data rev2.csv",
                    header = TRUE)
#  
# combine all data for plotting later
#
  All_data <- rbind(Sales, Future_Sales)
#
  linear_model <- lm(Y ~ a +
                     c +
                     f +
                     g +
                     h +
                     i +
                     j +
                     k,
                     data = Sales)
#
# normalize the training data for neural net model
#
  max_train <- vapply(Sales[1:nrow(Sales),], 
                FUN = max, 2)
  min_train <- vapply(Sales[1:nrow(Sales),], 
                FUN = min, 2)
#
  scaled_train <- as.data.frame(scale(Sales, 
                                     center = min_train, 
                                     scale = max_train
                                           - min_train))
#
# scale all data
#
  scaled_all <- as.data.frame((scale(All_data,
                                     center = min_train,
                                     scale = max_train
                                           - min_train)))
#
# train a neural network using neuralnet package
#
# choose an algorithm
#
  algorithms <- c("backprop","rprop+","rprop-","sag","slr")
#
# let 1 == "backprop"
#     2 == "rprop+"
#     3 == "rprop"
#     4 == "sag"
#     5 == "slr"
#
  use_algorithm <- algorithms[2]
#
# learning rate is only used with traditional gradient descent
#
  if (use_algorithm != "backprop") {
    learningrate <- NULL
  } else {
    learningrate <- 0.01
  }
#
# define layers
# note that if there is more than 1 hidden layer
# then layers_to_use must be a vector (e.g. c(7,3))
#
  layers_to_use <- c(7,5,3)
#
# initialize all the control variables
#
  use_threshold <- 0.001
#
# use_steps sets the maximum steps in the 
# greadient descent in function neuralnet
#
  use_steps <- 500000
#
# learningrate.factor controls what happens when
# the gradients are negative or positive, respectively
#
  learningrate.factor <- list(minus = 0.3, plus = 1.2)
#
# report_progress sets how many steps to go before
# reporting out convergence progress from function neuralnet
#
  report_progress <- 2000
#
# determine nodes in the netowrk
# note the following formula only works for 1 layer
#
  total_nodes <- nrow(scaled_train) 
  for (i in 1:length(layers_to_use)) {
    total_nodes <- total_nodes*(layers_to_use[i] + 1)
  }
#
# start weights are randomized by the algorithm
# the following does a search for "best" weights
# then those are passed to the algorithm
#
# create weights vector to send to neuralnet
#
  startweights_values <- vector(mode = "numeric", length = total_nodes)
#
# initialize the weights to random values from 0 to 1
#
  set.seed(Sys.time())
  startweights_values <- runif(length(startweights_values))
#
#
# repeat_random_process is used within function neuralnet
# to do a complete restart (random weights)
#
  repeat_random_process <- 1
#
  NN <- neuralnet(Y ~ a + c + f + g +
                      h + i + j +k,
                  scaled_train,
                  algorithm = use_algorithm,
                  threshold = use_threshold,
                  stepmax = use_steps,
                  lifesign.step = report_progress,
                  lifesign = "full",
                  hidden = layers_to_use,
                  startweights = startweights_values,
                  rep = repeat_random_process,
                  linear.output = TRUE,
                  learningrate = learningrate,
                  learningrate.factor = learningrate.factor)
#
# plot resulting network
#
  plot(NN,rep = "best")
#
# present results
#
# calculate linear predictions
#
  Z <- predict(linear_model, All_data)
#
# calculate linear residuals
#
  hist_values <- Z[1:nrow(Sales)] - Sales$Y
  x_min <- 1.1 * min(hist_values)
  x_max <- 1.1 * max(hist_values)
  if (abs(x_min) > abs(x_max)) {
    x_max = abs(x_min)
  } else {
    x_min = -1 * x_max
  }
#
# plot linear residuals hstogram
#
  hist(hist_values,
       breaks = seq(x_min, x_max, (x_max - x_min)/20),
       main = "Distribution of Residual Errors",
       xlab = "residual error",
       ylab = "count")
#
# calculate neural net predictions
#
# note that the second line below scales the predictions back
# to the orignal scale by applying the scale values from 
# the last column of data, which are the data we are trying to model
# i.e. max[length(scaled)] is the last value in the max vector
# and min[length(scaled)] is the last value in the min vector
#
  predict_NN_temp <- compute(NN, scaled_all[,c(1:8)])
  predict_NN <- predict_NN_temp$net.result *
                   (max_train[length(scaled_all)] - 
                    min_train[length(scaled_all)]) + 
                    min_train[length(scaled_all)]

# calculate residuals for neural net model
#
  hist_values <- predict_NN[1:nrow(Sales)] - Sales$Y
  x_min <- 1.1 * min(hist_values)
  x_max <- 1.1 * max(hist_values)
  if (abs(x_min) > abs(x_max)) {
    x_max = abs(x_min)
  } else {
    x_min = -1 * x_max
  }
#
# plot neural net residuals histogram
#
   hist(hist_values,
       breaks = seq(x_min, x_max, (x_max - x_min)/20),
       main = "Distribution of Residual Errors",
       xlab = "residual error",
       ylab = "count")
#
# plot just the data to be modeled
#
# plot training data
#
   plot(x = Sales$a, 
        y = Sales$Y, 
        type = "p", 
        col = "black",
        cex = 0.5,
        lwd = 0.25,
        main = "Training and Test Data",
        xlab = "Days",
        ylab = "Sales",
        yaxt = "n",
        xlim = c(min(Sales$a), max(Sales$a) + 90),
        ylim = c(0.9 * min(Sales$Y), 1.1 * max(Future_Sales$Y)))
#
   axis(2, at = axTicks(2),
        labels = sprintf("$%s", axTicks(2)))
#
# identify training range
#
   text(0.33 * (Sales$a[nrow(Sales)]
                -  Sales$a[1])
        +  Sales$a[1],
        0.50 * (Sales$Y[nrow(Sales)]
                -  Sales$Y[1])
        +  Sales$Y[1],
        pos = 3,
        "training data",
        col = "black")
#
# identify test range
#
   text(0.75 * (Sales$a[nrow(Sales)]
                -  Sales$a[1])
        +  Sales$a[1],
        1.10 * (Sales$Y[nrow(Sales)]
                -  Sales$Y[1])
        +  Sales$Y[1],
        pos = 3,
        "test data",
        col = "red")
#
# add test data to plot
#
   lines(x = Future_Sales$a,
         y = Future_Sales$Y,
         type = "p",
         col = "red",
         cex = 0.5,
         lwd = 0.25)
#
# plot results and underlying data
#
# plot training data
#
  plot(x = Sales$a, 
       y = Sales$Y, 
       type = "p", 
       col = "black",
       cex = 0.5,
       lwd = 0.25,
       main = "Data and Predictions",
       xlab = "Days",
       ylab = "Sales",
       yaxt = "n",
       xlim = c(min(Sales$a), max(Sales$a) + 90),
       ylim = c(0.9 * min(Sales$Y), 1.1 * max(Future_Sales$Y)))
#
  axis(2, at = axTicks(2),
       labels = sprintf("$%s", axTicks(2)))
  #
# identify training range
#
  text(0.33 * (Sales$a[nrow(Sales)]
            -  Sales$a[1])
            +  Sales$a[1],
       0.50 * (Sales$Y[nrow(Sales)]
            -  Sales$Y[1])
            +  Sales$Y[1],
       pos = 3,
       "training data",
       col = "black")
#
# identify test range
#
  text(0.75 * (Sales$a[nrow(Sales)]
            -  Sales$a[1])
            +  Sales$a[1],
       1.10 * (Sales$Y[nrow(Sales)]
            -  Sales$Y[1])
            +  Sales$Y[1],
       pos = 3,
       "test data",
       col = "red")
#
# add test data to plot
#
  lines(x = Future_Sales$a,
        y = Future_Sales$Y,
        type = "p",
        col = "red",
        cex = 0.5,
        lwd = 0.25)
#
# add linear predictions to plot
#
  lines(x = All_data$a, 
        y = Z, 
        type = "l", 
        col = "blue",
        lwd = 0.5)
#
# label linear predictions
#
  text(All_data$a[0.25*nrow(All_data)],
       Z[20],
       pos = 4,
       adj = 1,
       "predictions from linear model (lm())",
       col = "blue")
#
# add nn predictions to plot
#
  lines(x = All_data$a, 
        y = predict_NN,
        type = "l", 
        col = "green",
        lwd = 0.5)
#
# label neural net predictions
#
  text(0.25 * All_data$a[nrow(predict_NN)],
       predict_NN[1 * nrow(predict_NN)],
       pos = 4,
       "predictions from",
       col = "green")
#
  text(0.25 * All_data$a[nrow(predict_NN)],
       predict_NN[0.952 * nrow(predict_NN)],
       pos = 4,
       "non-linear model",
       col = "green")
#
  text(0.25 * All_data$a[nrow(predict_NN)],
       predict_NN[0.936 * nrow(predict_NN)],
       pos = 4,
       "(neuralnet())",
       col = "green")  
#
# plot only the test range
#
  y_range <- c(0.9 * min(Future_Sales$Y), 
               1.1 * max(Future_Sales$Y))
  x_range <- c(min(Future_Sales$a), max(Future_Sales$a))
#                      
  plot(x = Future_Sales$a, 
              y = Future_Sales$Y, 
              type = "p", 
              col = "red",
              cex = 1,
              lwd = 0.25,
              xlim = x_range,
              ylim = y_range,
              xlab = "Days",
              ylab = "Sales",
              yaxt = "n",
              main = "Data and Predictions")
#
#
  axis(2, at = axTicks(2),
       labels = sprintf("$%s", axTicks(2)))
#
# add linear predictions to plot
#
  lines(x = Future_Sales$a,
              y = Z[(nrow(Sales) + 1):(length(Z))],
              type = "l",
              col = "blue",
              lwd = 0.25)
#
# add neural net predictions to plot
#
  lines(x = All_data$a[(nrow(Sales) + 1):(nrow(predict_NN))], 
        y = predict_NN[(nrow(Sales) + 1):(nrow(predict_NN))],
        type = "l", 
        col = "green",
        lwd = 0.25)
#
# label test data
#
  text(0.10 * (x_range[2] - x_range[1]) + x_range[1],
       0.50 * (y_range[2] - y_range[1]) + y_range[1],
       pos = 4,
       "test data",
       col = "red")
#
# label linear predictions
#
  text(0.50 * (x_range[2] - x_range[1]) + x_range[1],
       0.20 * (y_range[2] - y_range[1]) + y_range[1],
       pos = 4,
       "predictions from",
       col = "blue")
#
  text(0.50 * (x_range[2] - x_range[1]) + x_range[1],
       0.15 * (y_range[2] - y_range[1]) + y_range[1],
       pos = 4,
       "linear model (lm())",
       col = "blue")
#
# label neural net predictions
#
  text(0.60 * (x_range[2] - x_range[1]) + x_range[1],
       0.85 * (y_range[2] - y_range[1]) + y_range[1],
       pos = 4,
       "predictions from",
       col = "green")
#
  text(0.60 * (x_range[2] - x_range[1]) + x_range[1],
       0.80 * (y_range[2] - y_range[1]) + y_range[1],
       pos = 4,
       "non_linear model",
       col = "green")
#
  text(0.60 * (x_range[2] - x_range[1]) + x_range[1],
       0.75 * (y_range[2] - y_range[1]) + y_range[1],
       pos = 4,
       "(neuralnet())",
       col = "green")  
#
  