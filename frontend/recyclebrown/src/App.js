import styles from './App.module.css';
import Navbar from './Components/Navbar/Navbar';
import Introduction from './Components/Introduction/Introduction';
import Detector from "./Components/Detector/Detector";
import Modal from './Components/Modal/Modal';
import Footer from './Components/Footer/Footer';
import 'reactjs-popup/dist/index.css';
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

function App() {

  const [isOpen, setIsOpen] = useState(false)
  const [category, setCategory] = useState(10)

  const [imgFile, setImgFile] = useState(false)

  const url = {
    model: '../models/model.json',
  };

  async function loadModel(url) {
    try {
      const model = await tf.loadGraphModel(url.model);
      setModel(model);
      console.log("Load model success")
    }
    catch (err) {
      console.log(err);
    }
  }

  const [model, setModel] = useState();
  useEffect(() => {
    tf.ready().then(() => {
    loadModel(url)
   });
  },[])


  function makePrediction() {
    let res = 0;
    res = model.predict(imgFile)
    console.log(res)
    setIsOpen(!isOpen)
    setCategory(res)
  }

  function modalToggler() {
    setIsOpen(!isOpen)
  }

  function runModalHandler(e) {
    console.log("GO");
  };

  return (
    <div className={styles.App}>
      <Navbar />
      <Introduction />
      <Detector toggler={makePrediction} img={imgFile} setImg={setImgFile}/>
      {isOpen && <Modal name={category} toggler={modalToggler}/>}
      <Footer />
    </div>
  );
}

export default App;