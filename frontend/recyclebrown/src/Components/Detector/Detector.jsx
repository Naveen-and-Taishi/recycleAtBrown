import React, {useState, useCallback, useMemo} from 'react';
import styles from "./Detector.module.css";
import ImageCapture from 'react-image-data-capture';

function Detector(props) {

    const [imgSrc, setImgSrc] = useState(null);

    const onCapture = (imageData) => {
        setImgSrc(imageData.webP);
        props.setImg(imageData.file);
    };

  const onError = useCallback((error) => { console.log(error) }, []);
  const config = useMemo(() => ({ video: true }), []);
  const formData = new FormData();
  formData.append("file", props.img);

    return (
    <div className={styles.container}>
        <div className={styles.imageCap}>
            <h1>Try it out</h1>
            <ImageCapture
                className={styles.imageCap}
                onCapture={onCapture}
                onError={onError}
                width={500}
                userMediaConfig={config}
            />
        </div>
        <div className={styles.gap}>
            <p>fldjksffdafdsafasfs</p>
        </div>
        <div className={styles.yourPhotos}>
            <h1>Your Photo</h1>
            <div className={styles.photoContainer}>
                {imgSrc &&
                    // <div className={styles.imageBox}>
                    //     <img className={styles.image} src={imgSrc} alt="captured-img" />
                    // </div>
                    <img className={styles.image} src={imgSrc} alt="captured-img" />
                }
                
            </div>
            <button className={styles.button} onClick={props.toggler}>Go!</button>
        </div>
       
    </div>);
}

export default Detector;