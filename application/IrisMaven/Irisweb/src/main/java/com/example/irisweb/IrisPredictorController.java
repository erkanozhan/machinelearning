package com.example.irisweb;

import org.springframework.web.bind.annotation.*;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.ObjectInputStream;
import java.util.ArrayList;

@RestController
@RequestMapping("/api/irisweb")
public class IrisPredictorController {

    private Classifier model;

    // Constructor: Modeli yükler
    public IrisPredictorController() {
        try (ObjectInputStream ois = new ObjectInputStream(getClass().getResourceAsStream("/MLP_iris_model.model"))) {
            model = (Classifier) ois.readObject();
            System.out.println("Model başarıyla yüklendi.");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Model yüklenirken bir hata oluştu.");
        }
    }

    // Web formdan gelen veriyi işler ve tahmin yapar
    @PostMapping("/predict")
    public String predict(
            @RequestParam("sepalLength") double sepalLength,
            @RequestParam("sepalWidth") double sepalWidth,
            @RequestParam("petalLength") double petalLength,
            @RequestParam("petalWidth") double petalWidth
    ) {
        try {
            // WEKA veri kümesini hazırla
            Instances dataset = new Instances("IrisDataset", getAttributes(), 0);
            dataset.setClassIndex(dataset.numAttributes() - 1);

            // Kullanıcıdan gelen verilere göre bir örnek oluştur
            Instance instance = new DenseInstance(dataset.numAttributes());
            instance.setDataset(dataset);

            instance.setValue(0, sepalLength);
            instance.setValue(1, sepalWidth);
            instance.setValue(2, petalLength);
            instance.setValue(3, petalWidth);

            // Model ile tahmin yap
            double result = model.classifyInstance(instance);
            return dataset.classAttribute().value((int) result);
        } catch (Exception e) {
            return "Tahmin sırasında hata oluştu: " + e.getMessage();
        }
    }

    private ArrayList<Attribute> getAttributes() {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("sepalLength"));
        attributes.add(new Attribute("sepalWidth"));
        attributes.add(new Attribute("petalLength"));
        attributes.add(new Attribute("petalWidth"));

        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("Iris-setosa");
        classValues.add("Iris-versicolor");
        classValues.add("Iris-virginica");
        attributes.add(new Attribute("class", classValues));
        return attributes;
    }
}
