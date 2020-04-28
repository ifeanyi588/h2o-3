package hex.tree.xgboost.util;

import ml.dmlc.xgboost4j.java.*;
import org.apache.log4j.Logger;
import water.util.Log;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Utility to access package private Booster methods.
 */
public class BoosterHelper {

  private static final Logger LOG = Logger.getLogger(BoosterHelper.class);


  public static Booster loadModel(InputStream in) throws XGBoostError, IOException {
    return XGBoost.loadModel(in);
  }

  public static Booster loadModel(byte[] boosterBytes) {
    if (boosterBytes == null) {
      throw new IllegalArgumentException("Booster not initialized!");
    }
    try {
      Booster booster = XGBoost.loadModel(new ByteArrayInputStream(boosterBytes));
      LOG.debug("Booster created from bytes, raw size = " + boosterBytes.length);
      return booster;
    } catch (XGBoostError | IOException exception) {
      throw new IllegalStateException("Failed to load the booster.", exception);
    }
  }

  /**
   * Invalidates XGBoost objects (Booster, DMatrix) and frees up their memory
   * @param xgbObjects list of XGBoost objects
   * @throws IllegalStateException when object invalidation fails, only the first exception will be reported (as the
   * exception cause), we assume the other ones will have a same reason
   */
  public static void dispose(Object... xgbObjects) throws IllegalStateException {
    Exception firstException = null;
    for (Object xgbObject : xgbObjects) {
      if (xgbObject == null)
        continue;
      if (xgbObject instanceof Booster) {
        try {
          ((Booster) xgbObject).dispose();
        } catch (Exception e) {
          if (firstException == null)
            firstException = e;
        }
      } else if (xgbObject instanceof DMatrix) {
        try {
          ((DMatrix) xgbObject).dispose();
        } catch (Exception e) {
          if (firstException == null)
            firstException = e;
        }
      } else
        assert false : "Unsupported XGBoost object type: " + xgbObject.getClass();
    }
    if (firstException != null) {
      throw new IllegalStateException("We were unable to free-up xgboost memory. " +
              "This could indicate a memory leak and it can lead to H2O instability.", firstException);
    }
  }

  public interface BoosterOp<X> {
    X apply(Booster booster) throws XGBoostError;
  }

  public static <X> X doWithLocalRabit(BoosterOp<X> op, Booster booster) throws XGBoostError {
    boolean shutdownRabit = true;
    try {
      Map<String, String> rabitEnv = new HashMap<>();
      rabitEnv.put("DMLC_TASK_ID", "0");
      Rabit.init(rabitEnv);
      shutdownRabit = true;
      X result = op.apply(booster);
      Rabit.shutdown();
      shutdownRabit = false;
      return result;
    } finally {
      if (shutdownRabit)
        try {
          Rabit.shutdown();
        } catch (XGBoostError e) {
          e.printStackTrace(); // don't rely on logging support in genmodel
        }
    }
  }

}
