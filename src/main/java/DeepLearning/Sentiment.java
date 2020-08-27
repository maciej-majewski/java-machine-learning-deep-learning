package DeepLearning;

public class Sentiment {
    private Double topic1Sentiment;
    private Double topic2Sentiment;
    private Double topic3Sentiment;
    private Double topic4Sentiment;

    private String sentimentClass;

    public Sentiment(Double topic1Sentiment, Double topic2Sentiment, Double topic3Sentiment, Double topic4Sentiment) {
        this.topic1Sentiment = topic1Sentiment;
        this.topic2Sentiment = topic2Sentiment;
        this.topic3Sentiment = topic3Sentiment;
        this.topic4Sentiment = topic4Sentiment;
    }

    public Double getTopic1Sentiment() {
        return topic1Sentiment;
    }

    public void setTopic1Sentiment(Double topic1Sentiment) {
        this.topic1Sentiment = topic1Sentiment;
    }

    public Double getTopic2Sentiment() {
        return topic2Sentiment;
    }

    public void setTopic2Sentiment(Double topic2Sentiment) {
        this.topic2Sentiment = topic2Sentiment;
    }

    public Double getTopic3Sentiment() {
        return topic3Sentiment;
    }

    public void setTopic3Sentiment(Double topic3Sentiment) {
        this.topic3Sentiment = topic3Sentiment;
    }

    public Double getTopic4Sentiment() {
        return topic4Sentiment;
    }

    public void setTopic4Sentiment(Double topic4Sentiment) {
        this.topic4Sentiment = topic4Sentiment;
    }

    public String getSentimentClass() {
        return sentimentClass;
    }

    public void setSentimentClass(String sentimentClass) {
        this.sentimentClass = sentimentClass;
    }

    @Override
    public String toString() {
        return String.format(
                "Sentiment class = %s, Data[ Topic#1 Sentiment = %.1f, Topic#2 Sentiment = %.1f, Topic#3 Sentiment = %.1f, Topic#4 Sentiment = %.1f ]",
                sentimentClass, topic1Sentiment, topic2Sentiment, topic3Sentiment, topic4Sentiment);
    }

}
