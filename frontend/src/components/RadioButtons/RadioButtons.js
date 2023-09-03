import "./RadioButtons.css";
import { useStore } from "../../helpers/store";

function RadioButtons() {
  const preprocessing = useStore((state) => state.preprocessing);
  const weight = useStore((state) => state.weight);
  const setModel = useStore((state) => state.setModel);
  const setPreprocessing = useStore((state) => state.setPreprocessing);
  const setWeight = useStore((state) => state.setWeight);

  const onChangeModel = (e) => {
    console.log(e.target.value);
    if (e.target.value === "tf" || e.target.value === "tf-idf") {
      setPreprocessing(true);
      setWeight(false);
    } else if (e.target.value === "custom" || e.target.value === "pretrained") {
      setPreprocessing(false);
      setWeight(true);
    } else {
      setPreprocessing(false);
      setWeight(false);
    }
    setModel(e.target.value);
  };

  const onChangePreprocessing = (e) => {
    console.log(e.target.value);
    setPreprocessing(e.target.value);
  };

  const onChangeWeight = (e) => {
    console.log(e.target.value);
    setWeight(e.target.value);
  };

  return (
    <div className="w-25 px-3">
      <div onChange={onChangeModel}>
        <h5>Model</h5>
        <div className="form-check">
          <input
            className="form-check-input"
            type="radio"
            name="exampleRadios"
            value="custom"
            defaultChecked
          />
          <label className="form-check-label">Custom word vectors</label>
        </div>
        <div className="form-check">
          <input
            className="form-check-input"
            type="radio"
            name="exampleRadios"
            value="pretrained"
          />
          <label className="form-check-label">Pretrained word vectors</label>
        </div>
        <div className="form-check">
          <input
            className="form-check-input"
            type="radio"
            name="exampleRadios"
            value="tourbert"
          />
          <label className="form-check-label">TourBERT word vectors</label>
        </div>
        <div className="form-check">
          <input
            className="form-check-input"
            type="radio"
            name="exampleRadios"
            value="tf"
          />
          <label className="form-check-label">TF</label>
        </div>
        <div className="form-check">
          <input
            className="form-check-input"
            type="radio"
            name="exampleRadios"
            value="tf-idf"
          />
          <label className="form-check-label">TF-IDF</label>
        </div>
      </div>
      {preprocessing && (
        <div className="px-3 pt-3" onChange={onChangePreprocessing}>
          <h5>Preprocessing</h5>
          <div className="form-check">
            <input
              className="form-check-input"
              type="radio"
              name="exampleRadios2"
              value="stemming"
              defaultChecked
            />
            <label className="form-check-label">Stemming</label>
          </div>
          <div className="form-check">
            <input
              className="form-check-input"
              type="radio"
              name="exampleRadios2"
              value="lemmatization"
            />
            <label className="form-check-label">Lemmatization</label>
          </div>
        </div>
      )}
      {weight && (
        <div className="px-3 pt-3" onChange={onChangeWeight}>
          <h5>Word vectors weight</h5>
          <div className="form-check">
            <input
              className="form-check-input"
              type="radio"
              name="exampleRadios3"
              value="None"
              defaultChecked
            />
            <label className="form-check-label">None</label>
          </div>
          <div className="form-check">
            <input
              className="form-check-input"
              type="radio"
              name="exampleRadios3"
              value="idf"
            />
            <label className="form-check-label">IDF</label>
          </div>
          <div className="form-check">
            <input
              className="form-check-input"
              type="radio"
              name="exampleRadios3"
              value="pos"
            />
            <label className="form-check-label">POS</label>
          </div>
          <div className="form-check">
            <input
              className="form-check-input"
              type="radio"
              name="exampleRadios3"
              value="ner"
            />
            <label className="form-check-label">NER</label>
          </div>
          <div className="form-check">
            <input
              className="form-check-input"
              type="radio"
              name="exampleRadios3"
              value="pos+ner"
            />
            <label className="form-check-label">POS+NER</label>
          </div>
        </div>
      )}
    </div>
  );
}

export default RadioButtons;
