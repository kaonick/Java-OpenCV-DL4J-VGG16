package com.ck.dl4j.vgg16;

import com.ck.cv.utils.Utils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils.ImageNetLabels;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;



/**
 * The class is reference the Myrobotlab DL4J filter
 *
 * @author <a href="mailto:kaonick@fpg.com.tw">Nick Kao</a>
 * @version 1.0 (2017-08-25)
 * @since 1.0 (2017-08-25)
 *
 */
public class OpenCVFilterDL4J implements Runnable {

  private static final long serialVersionUID = 1L;
  public final static Logger log = LoggerFactory.getLogger(OpenCVFilterDL4J.class.getCanonicalName());

  private ComputationGraph  netvgg16;
  private boolean keyForRun=true;//for check thread stop or not
  private int fontFace = Core.FONT_HERSHEY_PLAIN;
  private double fontScale = 1;
  private Scalar fontColor = new Scalar(0,0,255) ;;

  
  public Map<String, Double> lastResult = null;
  private Mat lastImage = null;
  
  public OpenCVFilterDL4J() {
    super();
    loadDL4j();
  }

  private void loadDL4j() {

    log.info("Loading VGG 16 Model.");

    ZooModel zooModel = new VGG16();
    try{
      netvgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
    }catch (Exception ex){

    }

    log.info("Done loading model..");
    
    // start classifier thread
    Thread classifier = new Thread(this, "DL4JClassifierThread");
    classifier.start();
  }
  

  /**
   * Method for process get print lastResult in image,and replace image to classify
   *
   * @param image
   *            one image for classify
   * @return the {@link Mat} to show
   */
  public Mat process(Mat image) throws InterruptedException {
    
    if (lastResult != null) {
      // the thread running will be updating lastResult for it as fast as it can.
      // log.info("Display result " );
      displayResult(image, lastResult);
    }
    // ok now we just need to update the image that the current thread is processing (if the current thread is idle i guess?)
    lastImage = image;
    return image;
  }


  /**
   * Method for stop the thread of DL4J Model
   */
  public void shutdown() {
    keyForRun=false;
  }
/**
  private static String padRight(String s, int n) {
    return String.format("%1$-" + n + "s", s);  
  }
*/

  private void displayResult(Mat image, Map<String, Double> result) {
    DecimalFormat df2 = new DecimalFormat("#.###");
    int i = 0;
    int percentOffset = 150;
    for (String label : result.keySet()) {
      i++;
      String val = df2.format(result.get(label)*100) + "%";
      Imgproc.putText(image,label + " : " , new Point(20, 60+(i*12)), fontFace, fontScale,fontColor);
      Imgproc.putText(image,val , new Point(20+percentOffset, 60+(i*12)), fontFace, fontScale,fontColor);

    }
  }

  private String formatResultString(Map<String, Double> result) {
    DecimalFormat df2 = new DecimalFormat("#.###");
    StringBuilder res = new StringBuilder();
    for (String key : result.keySet()) {
      res.append(key + " : ");
      res.append(df2.format(result.get(key)*100) + "% , ");        
    }
    return res.toString();
  }



  @Override
  public void run() {
    
    log.info("Starting the DL4J classifier thread...");
    // in a loop, grab the current image and classify it and update the result.
    while (true) {
      // log.info("Running!!!");
      // now we need to know which image we should classify
      // there likely needs to be some synchronization on this too.. o/w the main thread will
      // be updating it while it's being classified maybe?!
      if(keyForRun==false)break;


      if (lastImage != null) {
        try {
          lastResult = classify(lastImage);
          log.info(formatResultString(lastResult));
        } catch (Exception e) {
          // TODO Auto-generated catch block
          log.warn("Exception classifying image!");
          e.printStackTrace();
        }
      } else {
        // log.info("No Image to classify...");
      }
      // TODO: see why there's a race condition. i seem to need a little delay here o/w the recognition never seems to start.
      // maybe lastImage needs to be marked as volitite?
      try {
        Thread.sleep(1);
      } catch (InterruptedException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
    }
  }


  private Map<String, Double> classify(Mat frame){
    long begin=System.currentTimeMillis();
    NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
    ;
    INDArray image = null;
    try {
      image = loader.asMatrix(Utils.matToBufferedImage(frame));
    } catch (IOException e) {
      e.printStackTrace();
    }
    long now1=System.currentTimeMillis();
    System.out.println("time1="+(now1-begin));

    // Mean subtraction pre-processing step for VGG
    DataNormalization scaler = new VGG16ImagePreProcessor();
    scaler.transform(image);

    long now2=System.currentTimeMillis();
    System.out.println("time2="+(now2-now1));

    //Inference returns array of INDArray, index[0] has the predictions
    INDArray[] output = netvgg16.output(false,image);
    //INDArray output = netvgg16.outputSingle(image);
    long now3=System.currentTimeMillis();
    System.out.println("time3="+(now3-now2));

    // convert 1000 length numeric index of probabilities per label
    // to sorted return top 5 convert to string using helper function VGG16.decodePredictions
    // "predictions" is string of our results
    return decodeVGG16Predictions(output[0]);
  }

  private Map<String, Double> decodeVGG16Predictions(INDArray predictions) {

    LinkedHashMap<String, Double> recognizedObjects = new LinkedHashMap<String, Double>();
    ArrayList<String> labels;
    String predictionDescription = "";
    int[] top5 = new int[5];
    float[] top5Prob = new float[5];
    labels = ImageNetLabels.getLabels();
    //brute force collect top 5
    int i = 0;
    for (int batch = 0; batch < predictions.size(0); batch++) {
      if (predictions.size(0) > 1) {
        predictionDescription += String.valueOf(batch);
      }
      predictionDescription += " :";
      INDArray currentBatch = predictions.getRow(batch).dup();
      while (i < 5) {
        top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
        top5Prob[i] = currentBatch.getFloat(batch, top5[i]);
        // interesting, this cast looses precision.. float to double.
        recognizedObjects.put(labels.get(top5[i]), (double)top5Prob[i]);
        currentBatch.putScalar(0, top5[i], 0);
        predictionDescription += "\n\t" + String.format("%3f", top5Prob[i] * 100) + "%, " + labels.get(top5[i]);
        i++;
      }
    }
    return recognizedObjects;
  }


}
