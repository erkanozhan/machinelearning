import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;

public class IrisPredictorForm {
    private JPanel mainPanel;  // Sevgili gençler Ana panel, Swing bileşenlerini tutar
    private JTextField textFieldSepalLength; // Bu alan Sepal uzunluğunu girmek içindir.
    private JTextField textFieldSepalWidth;  // Buraya Sepal genişliği değerini yazabilirsiniz.
    private JTextField textFieldPetalLength; // Petal uzunluğunu girmek için bu alanı kullanabilirsiniz.
    private JTextField textFieldPetalWidth;  // Petal genişliğini buraya yazınız.
    private JButton predictButton;           // Tahmin işlemi için butona tıklayın.
    private JTextArea resultArea;            // Tahmin sonucunu bu alanda göreceksiniz.
    private JLabel SepalLength;              // Sepal uzunluğu için etiket

    private Classifier model;                // WEKA model nesnesi

    // Constructor: Formun başlangıç ayarlarını içerir
    public IrisPredictorForm() {
        try {
            // Modeli JAR dosyasından yükle
            InputStream modelStream = getClass().getResourceAsStream("/resources/MLP_iris_model.model");
            if (modelStream == null) {
                // Eğer model dosyası bulunamazsa kullanıcıya mesaj göster
                JOptionPane.showMessageDialog(null, "Model dosyası bulunamadı.");
                return;
            }
            // Model dosyasını yüklemek için ObjectInputStream kullanılır
            ObjectInputStream ois = new ObjectInputStream(modelStream);
            model = (Classifier) ois.readObject(); // Model dosyasını Classifier nesnesine çevir
            ois.close();
        } catch (Exception e) {
            // Model yüklenirken hata oluşursa kullanıcı bilgilendirilir
            JOptionPane.showMessageDialog(null, "Model yüklenirken hata oluştu: " + e.getMessage());
        }

        // Butona tıklama olayını ekle
        predictButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    // Gençler, buradan girilen değerleri okuyarak işlemlere başlıyoruz.
                    double sepalLength = Double.parseDouble(textFieldSepalLength.getText());
                    double sepalWidth = Double.parseDouble(textFieldSepalWidth.getText());
                    double petalLength = Double.parseDouble(textFieldPetalLength.getText());
                    double petalWidth = Double.parseDouble(textFieldPetalWidth.getText());

                    // WEKA'nın Instances nesnesini oluştur
                    Instances dataset = new Instances("IrisDataset", getAttributes(), 0);
                    dataset.setClassIndex(dataset.numAttributes() - 1); // Sınıf etiketi son sütun olarak ayarlanır

                    // Kullanıcıdan alınan değerlere uygun bir örnek (Instance) oluştur
                    Instance instance = new DenseInstance(dataset.numAttributes());
                    instance.setDataset(dataset); // Örneği veri kümesine bağla

                    // Özellik değerlerini örneğe ekle
                    instance.setValue(0, sepalLength); // Sepal uzunluğu
                    instance.setValue(1, sepalWidth);  // Sepal genişliği
                    instance.setValue(2, petalLength); // Petal uzunluğu
                    instance.setValue(3, petalWidth);  // Petal genişliği

                    // Modeli kullanarak tahmin yap
                    double result = model.classifyInstance(instance);

                    // Tahmin edilen sınıfı al ve sonucu kullanıcıya göster
                    resultArea.setText("Tahmin edilen sınıf: " + dataset.classAttribute().value((int) result));
                } catch (Exception ex) {
                    // Tahmin sırasında hata olursa kullanıcıya hata mesajı göster
                    JOptionPane.showMessageDialog(null, "Tahmin sırasında hata oluştu: " + ex.getMessage());
                }
            }
        });
    }

    // Bu metod, modelin ihtiyaç duyduğu özelliklerin yapılandırmasını sağlar
    private ArrayList<weka.core.Attribute> getAttributes() {
        ArrayList<weka.core.Attribute> attributes = new ArrayList<>();

        // Sepal ve Petal özellikleri için nitelikler tanımlanır
        attributes.add(new weka.core.Attribute("sepalLength"));
        attributes.add(new weka.core.Attribute("sepalWidth"));
        attributes.add(new weka.core.Attribute("petalLength"));
        attributes.add(new weka.core.Attribute("petalWidth"));

        // Sınıf etiketleri (Iris türleri) belirlenir
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("Iris-setosa");
        classValues.add("Iris-versicolor");
        classValues.add("Iris-virginica");
        attributes.add(new weka.core.Attribute("class", classValues));

        return attributes; // Özellikler listesi döndürülür
    }

    // Ana metod: Swing arayüzünü başlatır
    public static void main(String[] args) {
        JFrame frame = new JFrame("Iris Tahmin Uygulaması"); // Uygulama penceresi oluşturulur
        frame.setContentPane(new IrisPredictorForm().mainPanel); // Ana panel pencereye eklenir
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Pencere kapanma davranışı ayarlanır
        frame.pack(); // Bileşenlerin boyutuna göre pencere ayarlanır
        frame.setVisible(true); // Pencere görünür yapılır
    }
}
