import React from 'react';
import styles from "./Modal.module.css";
import { RiCloseLine } from "react-icons/ri";
import Map from "../Map/Map";
//import PVDMap from "../../providenceMap.jpg";

function Modal(props) {
  const categoryNumber = props.name 
  let categoryName = "";
  let description = "";
  switch(categoryNumber) {
    /* 
    # 0 - cardboard
    # 1 - glass
    # 2 - metal
    # 3 - paper
    # 4 - plastic
    # 5 - trash
    */
    case 0: 
      categoryName = "Cardboard"
      description = "As long as your cardboard and paperboard is clean and dry, it should be placed in your recycle bin.  Wet or greasy cardboard like pizza boxes or fast food boxes are considered a contaminate and belong in the garbage.  Wet or contaminated items can jam sorting equipment and ruin good, clean material. (Homewood Disposal Service)."
    break;
    case 1: 
      categoryName = "Glass"
      description = "Glass is 100% recyclable and can be recycled endlessly without loss in quality or purity. Recycled glass containers are always needed because glass manufacturers require high-quality recycled container glass to meet market demands for new glass containers. (Glass Packaging Institute)."
    break;
    case 2: 
      categoryName = "Metal"
      description = "Metals are essential, versatile and can be used in a number of ways. Metals can be used for industrial purposes such as the manufacture of trucks, cars, airplanes, ships, and railways. They can also be used to manufacture domestic items such as cutlery, crockery and even in packaging. The good thing about metal recycling is that metal can be recycled over and over without altering its properties. (Conserve Energy Future)."
    break;
    case 3: 
      categoryName = "Paper"
      description = "The average household throws away 13,000 separate pieces of paper each year. Most is packaging and junk mail. Each ton (2000 pounds) of recycled paper can save 17 trees, 380 gallons of oil, three cubic yards of landfill space, 4000 kilowatts of energy, and 7000 gallons of water. This represents a 64% energy savings, a 58% water savings, and 60 pounds less of air pollution! (University of Southern Indiana)."
    break;
    case 4: 
      categoryName = "Plastic"
      description = "Plastic is one of the most popular and useful materials of modern times: we now use about 20 times more plastic than we did 50 years ago. Its popularity and widespread use is why handling it responsibly and correctly once it becomes waste is so vitally important. We can optimise the lifespan of plastics by re-using and recycling items as many times as possible. (RecycleNow)."
    break;
    case 5: 
      categoryName = "Trash"
      description = "Believe it or not, recycling the wrong materials can actually cause harm to the environment. First, non-recyclable materials have to get transferred from the recycling facility to the landfill. That's an extra trip that could have been avoided if the non-recyclable material had been put in the garbage to begin with. Also, food contamination is a major issue. Things like greasy pizza boxes and unwashed peanut butter jars can contaminate an entire load of recyclable material. (Vogel Disposal)."
    break;
  }

  return (
    <>
      <div className={styles.darkBG} onClick={props.toggler}></div>
      <div className={styles.popup}> 
        <div className={styles.modal}>
          <div className={styles.modalHeader}>
            <h1 className={styles.heading}>{categoryName}</h1>
          </div>
          <button className={styles.closeBtn} onClick={props.toggler}>
            <RiCloseLine style={{ marginBottom: "-3px" }} />
          </button>
          <div className={styles.container}>
            <div className={styles.blurb}>
              <p className={styles.text}>{description}</p>
            </div>
            <div className={styles.gap}>
            </div>
            <div className={styles.image}>
              <img className={styles.img} src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.fortunebuilders.com%2Fwp-content%2Fuploads%2F2015%2F03%2Fprovidence-neighborhoods.gif&f=1&nofb=1" alt="map of Providence"/>
            </div>
          </div>
        </div>
      </div>
    </>);
}

export default Modal;