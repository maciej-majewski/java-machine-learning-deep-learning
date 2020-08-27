package DeepLearning;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class SentimentClassifier {
    private static Map<Integer, String> classifiers;

    public SentimentClassifier() {
        classifiers = new HashMap<>();
        classifiers.put(0, "Sentiment Class #1");
        classifiers.put(1, "Sentiment Class #2");
        classifiers.put(2, "Sentiment Class #3");
    }

    public void classify(String sentimentDataTrainFile, String sentimentDataTestFile)
            throws FileNotFoundException, IOException, InterruptedException {

        int labelIndex = 4;
        int numClasses = 3;

        int batchSizeTraining = 9;
        DataSet trainingData = readCSVDataset(sentimentDataTrainFile, batchSizeTraining, labelIndex, numClasses);

        // shuffling the training data to avoid any impact of ordering
        trainingData.shuffle();

        int batchSizeTest = 3;
        DataSet testData = readCSVDataset(sentimentDataTestFile, batchSizeTest, labelIndex, numClasses);

        Map<Integer, Sentiment> sentimentSet = objectify(testData);
        sentimentSet.forEach((k, v) -> System.out.println("Index:" + k + " -> " + v));

        // normalizing the training data
        DataNormalization normalizer = new NormalizerStandardize();
        // collecting the statistics from the training data
        normalizer.fit(trainingData);

        // applying normalization to the training data
        normalizer.transform(trainingData);

        // applying normalization to the test data
        normalizer.transform(testData);

        int numInputs = 4;
        int outputNum = 3;
        int iterations = 500;
        long seed = 123;

        System.out.println("Building model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.01)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2,
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
                .backprop(true)
                .pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainingData);

        // evaluating the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());

        eval.eval(testData.getLabels(), output);

        System.out.println(eval.stats());

        System.out.println(output);

        classify(output,sentimentSet);

        sentimentSet.forEach((k, v) -> System.out.println("Index:" + k + " -> " + v));

    }

    public DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        return iterator.next();
    }

    private Map<Integer, Sentiment> objectify(DataSet testData) {
        Map<Integer, Sentiment> iSentimentSet = new HashMap<>();
        INDArray sentimentScores = testData.getFeatureMatrix();
        for (int i = 0; i < sentimentScores.rows(); i++) {
            INDArray score = sentimentScores.slice(i);
            Sentiment sentis = new Sentiment(score.getDouble(0), score.getDouble(1), score.getDouble(2), score.getDouble(3));
            iSentimentSet.put(i, sentis);
        }
        return iSentimentSet;
    }

    private void classify(INDArray output, Map<Integer, Sentiment> sentimentSet) {
        for (int i = 0; i < output.rows(); i++) {
            Sentiment sentis = sentimentSet.get(i);
            // setting the classification from the fitted results
            sentis.setSentimentClass(classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));
        }
    }

    private float[] getFloatArrayFromSlice(INDArray rowScore) {
        float[] result = new float[rowScore.columns()];
        for (int i = 0; i < rowScore.columns(); i++) {
            result[i] = rowScore.getFloat(i);
        }
        return result;
    }

    private static int maxIndex(float[] vals) {
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++) {
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}
