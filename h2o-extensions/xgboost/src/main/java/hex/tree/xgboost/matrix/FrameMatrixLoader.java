package hex.tree.xgboost.matrix;

import hex.tree.xgboost.XGBoostModel;
import hex.tree.xgboost.XGBoostUtils;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.java.XGBoostModelInfo;
import ml.dmlc.xgboost4j.java.XGBoostSetupTask;
import water.fvec.Frame;

public class FrameMatrixLoader extends MatrixLoader {

    private final XGBoostModelInfo _modelInfo;
    private final XGBoostModel.XGBoostParameters _parms;
    private final boolean _sparse;
    private final Frame _trainFrame;

    public FrameMatrixLoader(
        XGBoostModel model,
        XGBoostModel.XGBoostParameters parms,
        XGBoostSetupTask.FrameNodes train
    ) {
        _modelInfo = model.model_info();
        _parms = parms;
        _sparse = model._output._sparse;
        _trainFrame = train._fr;
    }

    @Override
    public DMatrix makeLocalMatrix() throws XGBoostError {
        return XGBoostUtils.convertFrameToDMatrix(
            _modelInfo.dataInfo(),
            _trainFrame,
            _parms._response_column,
            _parms._weights_column,
            _parms._offset_column,
            _sparse
        );
    }
}
