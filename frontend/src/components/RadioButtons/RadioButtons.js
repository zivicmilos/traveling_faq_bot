import "./RadioButtons.css";

function RadioButtons({ onChangeValue }) {
  return (
    <div className="w-25 px-3" onChange={onChangeValue}>
      <div className="form-check">
        <input
          className="form-check-input"
          type="radio"
          name="exampleRadios"
          value="pretrained"
          defaultChecked
        />
        <label className="form-check-label">Pretrained word vectors</label>
      </div>
      <div className="form-check">
        <input
          className="form-check-input"
          type="radio"
          name="exampleRadios"
          value="custom"
        />
        <label className="form-check-label">Custom word vectors</label>
      </div>
      <div className="form-check">
        <input
          className="form-check-input"
          type="radio"
          name="exampleRadios"
          value="finbert"
        />
        <label className="form-check-label">FinBERT word vectors</label>
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
  );
}

export default RadioButtons;
