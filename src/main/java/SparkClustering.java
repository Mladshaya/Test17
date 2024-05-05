import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SparkClustering {
    public static void main(String[] args) {
        Logger.getLogger("org.apache.spark").setLevel(Level.ERROR);

        // Создание SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("Spark Clustering")
                .config("spark.master", "local")
                .getOrCreate();

        // Источник данных: файл
        DataSource fileDataSource = new FileDataSource("C:\\Users\\Elena Shustova\\Desktop\\FOLDER\\SGTU\\DIPLOM\\iris.csv", spark);
        Dataset<Row> fileData = fileDataSource.getData();

        fileData.show();
        // Выбор признаков для кластеризации
        String[] selectedFeatures = new String[]{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"};

        // Выбор алгоритма кластеризации
        ClusterizerSelect clusterizerSelect = new ClusterizerSelect();
        String algorithm = "bisectingKMeans"; // Можно выбрать "kMeans" или "bisectingKMeans"
        Clusterizer<?> clusterizer = clusterizerSelect.createClusterizer(algorithm, selectedFeatures);

        // Применение выбранного алгоритма кластеризации к данным из файла
        Model<?> clusteringModel = clusterizer.cluster(fileData);

        // Создание экземпляра класса AnomalyDetector
        AnomalyDetector<?> anomalyDetector = new AnomalyDetector<>(clusteringModel, selectedFeatures);

        // Новые данные для обнаружения аномалий
        DataSource newDataSource = new FileDataSource("C:\\Users\\Elena Shustova\\Desktop\\FOLDER\\SGTU\\DIPLOM\\iris test.csv", spark);
        Dataset<Row> newData = newDataSource.getData();

        // Обнаружение аномалий на новых данных
        anomalyDetector.detectAnomalies(newData);

        // Закрытие SparkSession
        spark.stop();
    }
}

// Интерфейс для источника данных
interface DataSource {
    Dataset<Row> getData();
}

// Класс для работы с файловым источником данных
class FileDataSource implements DataSource {
    private final String filePath;
    private final SparkSession spark;

    FileDataSource(String filePath, SparkSession spark) {
        this.filePath = filePath;
        this.spark = spark;
    }

    @Override
    public Dataset<Row> getData() {
        // Чтение данных из файла и создание Dataset<Row> с автоматическим определением схемы
        Dataset<Row> data = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(filePath);

        // Индексирование строковых столбцов
        String[] stringColumns = Arrays.stream(data.columns())
                .filter(col -> data.schema().apply(col).dataType().typeName().equalsIgnoreCase("string"))
                .toArray(String[]::new);

        Dataset<Row> indexedData = data;

        // Индексация строковых столбцов
        for (String column : stringColumns) {
            StringIndexer stringIndexer = new StringIndexer()
                    .setInputCol(column)
                    .setOutputCol(column + "_index")
                    .setHandleInvalid("skip");

            indexedData = stringIndexer.fit(indexedData).transform(indexedData);
        }

        // Удаление строковых столбцов из итогового датасета
        for (String column : stringColumns) {
            indexedData = indexedData.drop(column);
        }

        // Возвращение итогового датасета
        return indexedData;
    }
}

// Интерфейс для кластеризации
interface Clusterizer<T extends Model<?>> {
    T cluster(Dataset<Row> data);
}

// Класс для кластеризации с использованием алгоритма KMeans
class KMeansClusterizer implements Clusterizer<KMeansModel> {
    private final String[] selectedFeatures;

    KMeansClusterizer(String[] selectedFeatures) {
        this.selectedFeatures = selectedFeatures;
    }


    @Override
    public KMeansModel cluster(Dataset<Row> data) {
        // Определение оптимального числа кластеров
        int optimalClusters = determineOptimalClusters(data);
        System.out.println("Оптимальное количество кластеров для KMeans: " + optimalClusters);

        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(selectedFeatures)
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(data);

        // Создание и обучение модели KMeans
        KMeans kmeans = new KMeans()
                .setK(optimalClusters)
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        KMeansModel model = kmeans.fit(featureData);

        // Вывод результатов кластеризации
        Dataset<Row> predictions = model.transform(featureData);
        System.out.println("Результаты кластеризации с KMeans:");
        predictions.show();

        return model;
    }

    // Метод для определения оптимального числа кластеров
    private int determineOptimalClusters(Dataset<Row> data) {
        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(selectedFeatures)
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(data);

        // Определение оптимального числа кластеров с помощью метрики силуэта
        int minClusters = 3;
        int maxClusters = 10;
        int optimalClusters = 0;
        double maxSilhouette = 0;

        for (int k = minClusters; k <= maxClusters; k++) {
            KMeans kmeans = new KMeans()
                    .setK(k)
                    .setSeed(1L)
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster");

            KMeansModel model = kmeans.fit(featureData);

            Dataset<Row> predictions = model.transform(featureData);

            ClusteringEvaluator evaluator = new ClusteringEvaluator()
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster")
                    .setMetricName("silhouette");

            double silhouette = evaluator.evaluate(predictions);
            System.out.println("Значение силуэта для KMeans с " + k + " кластерами: " + silhouette);
            if (silhouette > maxSilhouette) {
                maxSilhouette = silhouette;
                optimalClusters = k;
            }
        }

        return optimalClusters;
    }
}

// Класс для кластеризации с использованием алгоритма Bisecting KMeans
class BisectingKMeansClusterizer implements Clusterizer<BisectingKMeansModel> {
    private final String[] selectedFeatures;

    BisectingKMeansClusterizer(String[] selectedFeatures) {
        this.selectedFeatures = selectedFeatures;
    }

    @Override
    public BisectingKMeansModel cluster(Dataset<Row> data) {
        // Определение оптимального числа кластеров
        int optimalClusters = determineOptimalClusters(data);
        System.out.println("Оптимальное количество кластеров для Bisecting KMeans: " + optimalClusters);

        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(selectedFeatures)
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(data);

        // Создание и обучение модели Bisecting KMeans
        BisectingKMeans bkm = new BisectingKMeans()
                .setK(optimalClusters)
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        BisectingKMeansModel model = bkm.fit(featureData);

        // Вывод результатов кластеризации
        Dataset<Row> predictions = model.transform(featureData);
        System.out.println("Результаты кластеризации с Bisecting KMeans:");
        predictions.show();

        return model;
    }

    // Метод для определения оптимального числа кластеров
    private int determineOptimalClusters(Dataset<Row> data) {
        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(selectedFeatures)
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(data);

        // Определение оптимального числа кластеров с помощью метрики силуэта
        int minClusters = 3;
        int maxClusters = 10;
        int optimalClusters = 0;
        double maxSilhouette = 0;

        for (int k = minClusters; k <= maxClusters; k++) {
            BisectingKMeans bkm = new BisectingKMeans()
                    .setK(k)
                    .setSeed(1L)
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster");

            BisectingKMeansModel model = bkm.fit(featureData);

            Dataset<Row> predictions = model.transform(featureData);

            ClusteringEvaluator evaluator = new ClusteringEvaluator()
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster")
                    .setMetricName("silhouette");

            double silhouette = evaluator.evaluate(predictions);
            System.out.println("Значение силуэта для Bisecting KMeans с " + k + " кластерами: " + silhouette);
            if (silhouette > maxSilhouette) {
                maxSilhouette = silhouette;
                optimalClusters = k;
            }
        }

        return optimalClusters;
    }
}

//  Выбор и создание кластеризатора
class ClusterizerSelect {
    public Clusterizer<?> createClusterizer(String algorithm, String[] selectedFeatures) {
        if (algorithm.equalsIgnoreCase("kMeans")) {
            return new KMeansClusterizer(selectedFeatures);
        } else if (algorithm.equalsIgnoreCase("bisectingKMeans")) {
            return new BisectingKMeansClusterizer(selectedFeatures);
        } else {
            throw new IllegalArgumentException("Выбран неподдерживаемый алгоритм кластеризации.");
        }
    }
}

// Класс для обнаружения аномалий
class AnomalyDetector<T extends Model<? extends T>> {
    private final T clusteringModel;
    private final String[] selectedFeatures;

    public AnomalyDetector(T clusteringModel, String[] selectedFeatures) {
        this.clusteringModel = clusteringModel;
        this.selectedFeatures = selectedFeatures;
    }

    public void detectAnomalies(Dataset<Row> data) {
        if (clusteringModel == null) {
            throw new IllegalStateException("Модель кластеризации еще не обучена.");
        }

        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(selectedFeatures)
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(data);

        // Обработка аномалий в зависимости от типа модели кластеризации
        if (clusteringModel instanceof KMeansModel) {
            detectAnomaliesForKMeans((KMeansModel) clusteringModel, featureData);
        } else if (clusteringModel instanceof BisectingKMeansModel) {
            detectAnomaliesForBisectingKMeans((BisectingKMeansModel) clusteringModel, featureData);
        } else {
            throw new IllegalArgumentException("Неподдерживаемый тип модели кластеризации.");
        }
    }

    private void detectAnomaliesForKMeans(KMeansModel kmeansModel, Dataset<Row> data) {
        // Получение центров кластеров
        Vector[] centers = kmeansModel.clusterCenters();

        // Проход по данным и поиск аномалий для KMeans
        data.foreach(row -> {
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            // Вычисление расстояния до ближайшего центра кластера
            for (Vector center : centers) {
                double distance = Vectors.sqdist(features, center); // квадрат Евклидова расстояния
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            // Определение порогового значения для аномалий
            double anomalyThreshold = 2 * Math.sqrt(minDistance); // Примерный порог (можно настраивать)
            if (minDistance > anomalyThreshold) {
                String featuresString = Arrays.toString(features.toArray());
                String anomalyMessage = "Обнаружена аномалия в строке для KMeans: " + featuresString;
                System.out.println(anomalyMessage);
            }
        });
    }

    private void detectAnomaliesForBisectingKMeans(BisectingKMeansModel bkmModel, Dataset<Row> data) {
        // Получение центров кластеров
        Vector[] centers = bkmModel.clusterCenters();

        // Проход по данным и поиск аномалий для Bisecting KMeans
        data.foreach(row -> {
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            // Вычисление расстояния до ближайшего центра кластера
            for (Vector center : centers) {
                double distance = Vectors.sqdist(features, center); // квадрат Евклидова расстояния
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            // Определение порогового значения для аномалий
            double anomalyThreshold = 2 * Math.sqrt(minDistance); // Примерный порог (можно настраивать)
            if (minDistance > anomalyThreshold) {
                String featuresString = Arrays.toString(features.toArray());
                String anomalyMessage = "Обнаружена аномалия в строке для BisectingKMeans: " + featuresString;
                System.out.println(anomalyMessage);
            }
        });
    }
}

/*  //Попытка определения порога для обнаружения аномалий с помощью среднего и стандартного отклонения
class AnomalyDetector<T extends Model<? extends T>> {
    private final T clusteringModel;
    private final String[] selectedFeatures;
    private final double k; // Коэффициент для порога аномалий

    public AnomalyDetector(T clusteringModel, String[] selectedFeatures, double k) {
        this.clusteringModel = clusteringModel;
        this.selectedFeatures = selectedFeatures;
        this.k = k;
    }

    public void detectAnomalies(Dataset<Row> data) {
        if (clusteringModel == null) {
            throw new IllegalStateException("Модель кластеризации еще не обучена.");
        }

        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(selectedFeatures)
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(data);

        // Обработка аномалий в зависимости от типа модели кластеризации
        if (clusteringModel instanceof KMeansModel) {
            detectAnomaliesForKMeans((KMeansModel) clusteringModel, featureData);
        } else if (clusteringModel instanceof BisectingKMeansModel) {
            detectAnomaliesForBisectingKMeans((BisectingKMeansModel) clusteringModel, featureData);
        } else {
            throw new IllegalArgumentException("Неподдерживаемый тип модели кластеризации.");
        }
    }

    private void detectAnomaliesForKMeans(KMeansModel kmeansModel, Dataset<Row> data) {
        // Получение центров кластеров
        Vector[] centers = kmeansModel.clusterCenters();

        // Сбор расстояний до ближайшего центра кластера
        List<Double> distances = new ArrayList<>();
        data.foreach(row -> {
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            // Вычисление расстояния до ближайшего центра кластера
            for (Vector center : centers) {
                double distance = Vectors.sqdist(features, center); // квадрат Евклидова расстояния
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            distances.add(Math.sqrt(minDistance)); // Добавляем корень из квадрата расстояния
        });

        checkAnomalies(distances, k);
    }

    private void detectAnomaliesForBisectingKMeans(BisectingKMeansModel bkmModel, Dataset<Row> data) {
        // Получение центров кластеров
        Vector[] centers = bkmModel.clusterCenters();

        // Сбор расстояний до ближайшего центра кластера
        List<Double> distances = new ArrayList<>();
        data.foreach(row -> {
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            // Вычисление расстояния до ближайшего центра кластера
            for (Vector center : centers) {
                double distance = Vectors.sqdist(features, center); // квадрат Евклидова расстояния
                if (distance < minDistance) {
                    minDistance = distance;
                }
                System.out.println("Расстояние до центра кластера: " + distance);
            }
            distances.add(Math.sqrt(minDistance)); // Добавляем корень из квадрата расстояния
        });

        checkAnomalies(distances, k);
    }

    private double calculateMean(List<Double> distances) {
        double sumDistance = 0;
        for (double distance : distances) {
            sumDistance += distance;
        }
        return sumDistance / distances.size();
    }

    private double calculateStdDev(List<Double> distances, double meanDistance) {
        double sumSquaredDiff = 0;
        for (double distance : distances) {
            sumSquaredDiff += Math.pow(distance - meanDistance, 2);
        }
        return Math.sqrt(sumSquaredDiff / distances.size());
    }

    private double calculateAnomalyThreshold(List<Double> distances, double k) {
        double meanDistance = calculateMean(distances);
        double stdDevDistance = calculateStdDev(distances, meanDistance);
        return meanDistance + k * stdDevDistance;
    }

    private void checkAnomalies(List<Double> distances, double k) {
        double anomalyThreshold = calculateAnomalyThreshold(distances, k);
        for (double distance : distances) {
            System.out.println(distance);
            if (distance > anomalyThreshold) {
                String anomalyMessage = "Обнаружена аномалия: " + distance;
                System.out.println(anomalyMessage);
            }
        }
    }
}*/

