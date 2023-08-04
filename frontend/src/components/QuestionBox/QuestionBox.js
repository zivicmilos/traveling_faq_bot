import "./QuestionBox.css";
import { useEffect, useState } from "react";

function QuestionBox({  data, setData, onEnter }) {
  const handleChange = (e) => setData(e.target.value)

  return (
    <textarea
      className="form-control w-50 p-3"
      placeholder="Here you can input you question..."
      onChange={handleChange}
      value={data}
      onKeyDown={onEnter}
    ></textarea>
  );
}

export default QuestionBox;
