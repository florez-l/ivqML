// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__PCAImageFilter__h__
#define __ivqML__ITK__PCAImageFilter__h__

#include <itkImageToImageFilter.h>
#include <itkVectorImage.h>



#include <ivq/ITK/EigenUtils.h>
#include <ivqML/Common/PCA.h>





namespace ivqML
{
  namespace ITK
  {
    /**
     */
    template< class _TInImage, class _TReal = float >
    class PCAImageFilter
      : public itk::ImageToImageFilter< _TInImage, itk::VectorImage< _TReal, _TInImage::ImageDimension > >
    {
    public:
      using TInImage  = _TInImage;
      using TReal     = _TReal;
      using TOutImage = itk::VectorImage< TReal, TInImage::ImageDimension >;

      using Self         = PCAImageFilter;
      using Superclass   = itk::ImageToImageFilter< TInImage, TOutImage >;
      using Pointer      = itk::SmartPointer< Self >;
      using ConstPointer = itk::SmartPointer< const Self >;

    public:
      itkNewMacro( Self );
      itkTypeMacro(
        ivqML::ITK::PCAImageFilter, itk::ImageToImageFilter
        );

    protected:
      PCAImageFilter( )
        : Superclass( )
        {
        }
      virtual ~PCAImageFilter( ) override = default;

      /* TODO
         virtual void AllocateOutputs( ) override
         {
         }
      */

      virtual void GenerateOutputInformation( ) override
        {
          std::cout << "GenerateOutputInformation" << std::endl;

          const TInImage* in = this->GetInput( );
          TOutImage* out = this->GetOutput( );

          out->SetLargestPossibleRegion( in->GetLargestPossibleRegion( ) );
          out->SetRequestedRegion( in->GetRequestedRegion( ) );
          out->SetSpacing( in->GetSpacing( ) );
          out->SetOrigin( in->GetOrigin( ) );
          out->SetDirection( in->GetDirection( ) );
          out->SetNumberOfComponentsPerPixel( 10 );
        }

      virtual void GenerateData( ) override
        {
          std::cout << "GenerateData" << std::endl;

          const TInImage* in = this->GetInput( );

          auto X = ivq::ITK::ImageToMatrix( in ).transpose( );
          auto R = ivqML::Common::PCA< decltype( X ), TReal >( X, 0.95 );

          TOutImage* out = this->GetOutput( );
          out->SetNumberOfComponentsPerPixel( R.second.cols( ) );
          out->SetBufferedRegion( out->GetRequestedRegion( ) );
          out->Allocate( );
          std::cout << out->GetNumberOfComponentsPerPixel( ) << std::endl;

          ivq::ITK::ImageToMatrix( out ) = R.second.transpose( );
        }

    private:
      PCAImageFilter( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;
    };
  } // end namespace
} // end namespace

/* TODO
   #ifndef ITK_MANUAL_INSTANTIATION
   #  include <ivqML/ITK/PCAImageFilter.hxx>
   #endif // ITK_MANUAL_INSTANTIATION
*/

#endif // __ivqML__ITK__PCAImageFilter__h__

// eof - $RCSfile$
