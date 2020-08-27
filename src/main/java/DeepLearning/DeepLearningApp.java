package DeepLearning;

import java.io.IOException;

public class DeepLearningApp {
    public static void main(String[] args) throws IOException, InterruptedException {
        SentimentClassifier classifier = new SentimentClassifier();
        classifier.classify("sentiment.csv","sentiment-test.csv");
    }
}
