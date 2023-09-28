import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge as linearRegression
from sklearn.neighbors import KNeighborsRegressor as KNeighborsRegressor


def make_data(nSample, randomState, x0=None, std=0.5):
    np.random.seed(randomState)
    if x0 is None:
        X = np.sort(np.random.uniform(-10, 10, nSample))
    else:
        X = x0
    Y = np.sin(2 * X) + X * np.cos(X - 1) + np.random.normal(0, std, nSample)
    return X, Y


def estimateThings(nSample=1000, std=0.5, alphaValue=1.0, nNeighbors=5):
    dDataNbr = 100
    seed = 181683
    residual = np.empty(nSample)

    sampleX, sampleY = make_data(nSample=nSample,
                                 randomState=seed - 1,
                                 std=std)

    Y = np.empty((nSample, dDataNbr))
    X = np.empty((nSample, 1))
    trainedLinearY = np.empty((nSample, dDataNbr))
    trainedNonLinearY = np.empty((nSample, dDataNbr))

    for j in range(dDataNbr):
        X[:, 0], Y[:, j] = make_data(nSample, seed + j, sampleX, std)
        trainedLinearY[:, j] = linearRegression(alpha=alphaValue).fit(
            X, Y[:, j]).predict(X)
        trainedNonLinearY[:, j] = KNeighborsRegressor(nNeighbors).fit(
            X, Y[:, j]).predict(X)
    residual = np.var(Y, axis=1)

    linearVar = np.var(trainedLinearY, axis=1)
    linearBias = (np.mean(Y, axis=1) - np.mean(trainedLinearY, axis=1))**2

    expectedLinearError = residual + linearVar + linearBias

    nonLinearVar = np.var(trainedNonLinearY, axis=1)
    nonLinearBias = (np.mean(Y, axis=1) -
                     np.mean(trainedNonLinearY, axis=1))**2

    expectedNonLinearError = residual + nonLinearVar + nonLinearBias
    return residual, linearVar, linearBias, expectedLinearError, nonLinearVar, nonLinearBias, expectedNonLinearError, sampleX


if __name__ == "__main__":

    #Q2.d

    residual, linearVar, linearBias, expectedLinearError, nonLinearVar, nonLinearBias, expectedNonLinearError, sampleX = estimateThings(
    )

    plt.plot(sampleX, residual, linewidth=1)
    plt.savefig('residual.pdf')
    plt.close()

    plt.plot(sampleX, linearVar, linewidth=1)
    plt.savefig('linearVar.pdf')
    plt.close()
    plt.plot(sampleX, linearBias, linewidth=1)
    plt.savefig('linearBias.pdf')
    plt.close()
    plt.plot(sampleX, expectedLinearError, linewidth=1)
    plt.savefig('expectedLinearError.pdf')
    plt.close()

    plt.plot(sampleX, nonLinearVar, linewidth=1)
    plt.savefig('nonLinearVar.pdf')
    plt.close()
    plt.plot(sampleX, nonLinearBias, linewidth=1)
    plt.savefig('nonLinearBias.pdf')
    plt.close()
    plt.plot(sampleX, expectedNonLinearError, linewidth=1)
    plt.savefig('expectedNonLinearError.pdf')
    plt.close()

    #Q2.e

    nSamples = [10, 50, 100, 500, 1000]
    stds = [0.125, 0.25, 0.5, 0.75, 1]
    alphas = [0.25, 0.5, 0.75, 1, 2]
    nNeighbors = [1, 2, 5, 10, 50]

    residualMean = np.empty((5))
    linearVarMean = np.empty((5))
    linearBiasMean = np.empty((5))
    expectedLinearErrorMean = np.empty((5))
    nonLinearVarMean = np.empty((5))
    nonLinearBiasMean = np.empty((5))
    nonLinearExpectedErrorMean = np.empty((5))

    for i in range(len(nSamples)):
        residual, linearVar, linearBias, expectedLinearError, nonLinearVar, nonLinearBias, expectedNonLinearError, sampleX = estimateThings(
            nSample=nSamples[i])
        residualMean[i] = np.mean(residual)
        linearVarMean[i] = np.mean(linearVar)
        linearBiasMean[i] = np.mean(linearBias)
        expectedLinearErrorMean[i] = np.mean(expectedLinearError)
        nonLinearVarMean[i] = np.mean(nonLinearVar)
        nonLinearBiasMean[i] = np.mean(nonLinearBias)
        nonLinearExpectedErrorMean[i] = np.mean(expectedNonLinearError)

    plt.plot(nSamples, residualMean, linewidth=1)
    plt.savefig('residualnSamples.pdf')
    plt.close()

    plt.plot(nSamples, linearVarMean, linewidth=1)
    plt.savefig('linearVarnSamples.pdf')
    plt.close()
    plt.plot(nSamples, linearBiasMean, linewidth=1)
    plt.savefig('linearBiasnSamples.pdf')
    plt.close()
    plt.plot(nSamples, expectedLinearErrorMean, linewidth=1)
    plt.savefig('expectedLinearErrornSamples.pdf')
    plt.close()

    plt.plot(nSamples, nonLinearVarMean, linewidth=1)
    plt.savefig('nonLinearVarnSamples.pdf')
    plt.close()
    plt.plot(nSamples, nonLinearBiasMean, linewidth=1)
    plt.savefig('nonLinearBiasnSamples.pdf')
    plt.close()
    plt.plot(nSamples, nonLinearExpectedErrorMean, linewidth=1)
    plt.savefig('expectedNonLinearErrornSamples.pdf')
    plt.close()

    for i in range(len(stds)):
        residual, linearVar, linearBias, expectedLinearError, nonLinearVar, nonLinearBias, expectedNonLinearError, sampleX = estimateThings(
            std=stds[i])
        residualMean[i] = np.mean(residual)
        linearVarMean[i] = np.mean(linearVar)
        linearBiasMean[i] = np.mean(linearBias)
        expectedLinearErrorMean[i] = np.mean(expectedLinearError)
        nonLinearVarMean[i] = np.mean(nonLinearVar)
        nonLinearBiasMean[i] = np.mean(nonLinearBias)
        nonLinearExpectedErrorMean[i] = np.mean(expectedNonLinearError)

    plt.plot(stds, residualMean, linewidth=1)
    plt.savefig('residualstds.pdf')
    plt.close()

    plt.plot(stds, linearVarMean, linewidth=1)
    plt.savefig('linearVarstds.pdf')
    plt.close()
    plt.plot(stds, linearBiasMean, linewidth=1)
    plt.savefig('linearBiasstds.pdf')
    plt.close()
    plt.plot(stds, expectedLinearErrorMean, linewidth=1)
    plt.savefig('expectedLinearErrorstds.pdf')
    plt.close()

    plt.plot(stds, nonLinearVarMean, linewidth=1)
    plt.savefig('nonLinearVarstds.pdf')
    plt.close()
    plt.plot(stds, nonLinearBiasMean, linewidth=1)
    plt.savefig('nonLinearBiasstds.pdf')
    plt.close()
    plt.plot(stds, nonLinearExpectedErrorMean, linewidth=1)
    plt.savefig('expectedNonLinearErrorstds.pdf')
    plt.close()

    for i in range(len(nSamples)):
        residual, linearVar, linearBias, expectedLinearError, nonLinearVar, nonLinearBias, expectedNonLinearError, sampleX = estimateThings(
            alphaValue=alphas[i], nNeighbors=nNeighbors[i])
        residualMean[i] = np.mean(residual)
        linearVarMean[i] = np.mean(linearVar)
        linearBiasMean[i] = np.mean(linearBias)
        expectedLinearErrorMean[i] = np.mean(expectedLinearError)
        nonLinearVarMean[i] = np.mean(nonLinearVar)
        nonLinearBiasMean[i] = np.mean(nonLinearBias)
        nonLinearExpectedErrorMean[i] = np.mean(expectedNonLinearError)

    plt.plot(alphas, residualMean, linewidth=1)
    plt.savefig('residualcomplexity.pdf')
    plt.close()

    plt.plot(alphas, linearVarMean, linewidth=1)
    plt.savefig('linearVarcomplexity.pdf')
    plt.close()
    plt.plot(alphas, linearBiasMean, linewidth=1)
    plt.savefig('linearBiascomplexity.pdf')
    plt.close()
    plt.plot(alphas, expectedLinearErrorMean, linewidth=1)
    plt.savefig('expectedLinearErrorcomplexity.pdf')
    plt.close()

    plt.plot(nNeighbors, nonLinearVarMean, linewidth=1)
    plt.savefig('nonLinearVarcomplexity.pdf')
    plt.close()
    plt.plot(nNeighbors, nonLinearBiasMean, linewidth=1)
    plt.savefig('nonLinearBiascomplexity.pdf')
    plt.close()
    plt.plot(nNeighbors, nonLinearExpectedErrorMean, linewidth=1)
    plt.savefig('expectedNonLinearErrorcomplexity.pdf')
    plt.close()
