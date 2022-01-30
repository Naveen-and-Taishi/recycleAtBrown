import React, {useState, useCallback, useMemo} from 'react';
import styles from "./Detector.module.css";
import ImageCapture from 'react-image-data-capture';

function Detector(props) {

    const [imgSrc, setImgSrc] = useState(null);
    const [imgFile, setImgFile] = useState(null);
    const onCapture = (imageData) => {
        setImgSrc(imageData.webP);
        setImgFile(imageData.file);
    };

  const onError = useCallback((error) => { console.log(error) }, []);
  const config = useMemo(() => ({ video: true }), []);
  const formData = new FormData();
  formData.append("file", imgFile);

    return (
    <div className={styles.detector}>
        <>
        <ImageCapture
            onCapture={onCapture}
            onError={onError}
            width={500}
            userMediaConfig={config}
        />
        {imgSrc &&
            <div>
            <div>Captured Image:</div>
            <img src={imgSrc} alt="captured-img" />
            </div>
        }
        </>
    </div>);
}

export default Detector;
