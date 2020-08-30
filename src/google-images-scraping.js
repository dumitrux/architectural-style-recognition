    //https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/


    /**
     * simulate a right-click event so we can grab the image URL using the
     * context menu alleviating the need to navigate to another page
     *
     * attributed to @jmiserez: http://pyimg.co/9qe7y
     *
     * @param   {object}  element  DOM Element
     *
     * @return  {void}
     */
    function simulateRightClick( element ) {
        var event1 = new MouseEvent( 'mousedown', {
            bubbles: true,
            cancelable: false,
            view: window,
            button: 2,
            buttons: 2,
            clientX: element.getBoundingClientRect().x,
            clientY: element.getBoundingClientRect().y
        } );
        element.dispatchEvent( event1 );
        var event2 = new MouseEvent( 'mouseup', {
            bubbles: true,
            cancelable: false,
            view: window,
            button: 2,
            buttons: 0,
            clientX: element.getBoundingClientRect().x,
            clientY: element.getBoundingClientRect().y
        } );
        element.dispatchEvent( event2 );
        var event3 = new MouseEvent( 'contextmenu', {
            bubbles: true,
            cancelable: false,
            view: window,
            button: 2,
            buttons: 0,
            clientX: element.getBoundingClientRect().x,
            clientY: element.getBoundingClientRect().y
        } );
        element.dispatchEvent( event3 );
    }
    
    
    /**
     * grabs a URL Parameter from a query string because Google Images
     * stores the full image URL in a query parameter
     *
     * @param   {string}  queryString  The Query String
     * @param   {string}  key          The key to grab a value for
     *
     * @return  {string}               value
     */
    function getURLParam( queryString, key ) {
        var vars = queryString.replace( /^\?/, '' ).split( '&' );
        for ( let i = 0; i < vars.length; i++ ) {
            let pair = vars[ i ].split( '=' );
            if ( pair[0] == key ) {
                return pair[1];
            }
        }
        return false;
    }
    
    /**
     * Generate and automatically download a txt file from the URL contents
     *
     * @param   {string}  contents  The contents to download
     *
     * @return  {void}
     */
    function createDownload( contents ) {
        var hiddenElement = document.createElement( 'a' );
        hiddenElement.href = 'data:attachment/text,' + encodeURI( contents );
        hiddenElement.target = '_blank';
        //hiddenElement.download = 'urls.txt';
        hiddenElement.download = document.getElementById("REsRA").value + ".txt";
        hiddenElement.click();
    }
    
    
    /**
     * grab all URLs va a Promise that resolves once all URLs have been
     * acquired
     *
     * @return  {object}  Promise object
     */
    function grabUrls() {
        var urls = [];
        return new Promise( function( resolve, reject ) {
            var count = document.querySelectorAll(
                '.isv-r a:first-of-type' ).length,
                index = 0;
            Array.prototype.forEach.call( document.querySelectorAll(
                '.isv-r a:first-of-type' ), function( element ) {
                // using the right click menu Google will generate the
                // full-size URL; won't work in Internet Explorer
                // (http://pyimg.co/byukr)
                simulateRightClick( element.querySelector( ':scope img' ) );
                // Wait for it to appear on the <a> element
                var interval = setInterval( function() {
                    if ( element.href.trim() !== '' ) {
                        clearInterval( interval );
                        // extract the full-size version of the image
                        let googleUrl = element.href.replace( /.*(\?)/, '$1' ),
                            fullImageUrl = decodeURIComponent(
                                getURLParam( googleUrl, 'imgurl' ) );
                        if ( fullImageUrl !== 'false' ) {
                            urls.push( fullImageUrl );
                        }
                        // sometimes the URL returns a "false" string and
                        // we still want to count those so our Promise
                        // resolves
                        index++;
                        if ( index == ( count - 1 ) ) {
                            resolve( urls );
                        }
                    }
                }, 10 );
            } );
        } );
    }
    
    
    /**
     * Call the main function to grab the URLs and initiate the download
     */
    grabUrls().then( function( urls ) {
        urls = urls.join( '\n' );
        createDownload( urls );
    } );