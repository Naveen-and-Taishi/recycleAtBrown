import React from 'react';
import styles from "./Map.module.css";
// import {Map, InfoWindow, Marker, GoogleApiWrapper} from 'google-maps-react';
import {withScriptjs, withGoogleMap, GoogleMap} from "react-google-maps"

const Map = withScriptjs(withGoogleMap((props) =>
        <GoogleMap
            defaultZoom={7}
            defaultCenter={{ lat: 0, lng: 0}}
            ref={map => {
                const bounds = new window.google.maps.LatLngBounds();
                map && map.fitBounds(bounds)
            }}>
        </GoogleMap>
    ))

export default Map;