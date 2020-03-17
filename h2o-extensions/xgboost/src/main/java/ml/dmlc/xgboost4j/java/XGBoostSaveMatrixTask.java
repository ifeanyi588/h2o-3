package ml.dmlc.xgboost4j.java;

import org.apache.log4j.Logger;
import water.H2O;
import water.Key;

import java.io.File;
import java.io.IOException;

public class XGBoostSaveMatrixTask extends AbstractXGBoostTask<XGBoostSaveMatrixTask> {

    private static final Logger LOG = Logger.getLogger(XGBoostSaveMatrixTask.class);

    private final XGBoostMatrixFactory _matrixFactory;
    private final String _saveMatrixDirectory;
    
    protected transient DMatrix matrix;
    
    public XGBoostSaveMatrixTask(Key modelKey, String saveMatrixDirectory, boolean[] hasDMatrix, XGBoostMatrixFactory factory) {
        super(modelKey, hasDMatrix);
        _saveMatrixDirectory = saveMatrixDirectory;
        _matrixFactory = factory;
    }

    @Override
    protected void execute() {
        try {
            matrix = _matrixFactory.makeLocalMatrix();
            if (_saveMatrixDirectory != null) {
                File directory = new File(_saveMatrixDirectory);
                if (directory.mkdirs()) {
                    LOG.debug("Created directory for matrix export: " + directory.getAbsolutePath());
                }
                File path = new File(directory, "matrix.part" + H2O.SELF.index());
                LOG.info("Saving node-local portion of XGBoost training dataset to " + path.getAbsolutePath() + ".");
                matrix.saveBinary(path.getAbsolutePath());
            }
        } catch (XGBoostError| IOException xgBoostError) {
            throw new IllegalStateException("Failed XGBoost training.", xgBoostError);
        }
    }

}
