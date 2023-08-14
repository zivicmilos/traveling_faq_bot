import "./Navbar.css";
import { useStore } from "../../helpers/store";
import { useEffect } from "react";
import { storeData } from "../../helpers/local-storage";

function Navbar() {
  const handleReset = useStore((state) => state.handleReset);
  const defaultModel = useStore((state) => state.defaultModel);
  const setDefaultModel = useStore((state) => state.setDefaultModel);

  const handleModelType = (e) => {
    e.preventDefault();
    if (defaultModel === "default") setDefaultModel("customized");
    else if (defaultModel === "customized") setDefaultModel("similarity");
    else if (defaultModel === "similarity") setDefaultModel("table_qa");
    else if (defaultModel === "table_qa") setDefaultModel("gpt2");
    else if (defaultModel === "gpt2") setDefaultModel("bloom");
    else if (defaultModel === "bloom") setDefaultModel("chat_gpt");
    else if (defaultModel === "chat_gpt") setDefaultModel("default");
  };

  useEffect(() => {
    storeData("defaultModel", defaultModel);
  }, [defaultModel]);

  return (
    <nav className="fixed-top navbar navbar-expand-lg bg-body-tertiary">
      <div className="collapse navbar-collapse" id="navbarNav">
        <span className="navbar-brand mb-0 mx-3 h1">FAQ Bot</span>
        <ul className="navbar-nav">
          <li className="nav-item">
            <a className="nav-link" href="/" onClick={handleReset}>
              Reset chat
            </a>
          </li>
          <li className="nav-item">
            {defaultModel === "default" && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                Default model
              </a>
            )}
            {defaultModel === "customized" && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                Customize model
              </a>
            )}
            {defaultModel === "similarity" && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                Similarity model
              </a>
            )}
            {defaultModel === "table_qa" && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                Table QA model
              </a>
            )}
            {defaultModel === "gpt2" && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                GPT2 model
              </a>
            )}
            {defaultModel === "bloom" && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                Bloom model
              </a>
            )}
            {defaultModel === "chat_gpt" && (
              <a className="nav-link" href="/" onClick={handleModelType}>
                ChatGPT model
              </a>
            )}
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
