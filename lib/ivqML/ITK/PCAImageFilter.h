// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__PCAImageFilter__h__
#define __ivqML__ITK__PCAImageFilter__h__

#include <itkImageToImageFilter.h>
#include <itkVectorImage.h>



#include <itkGenerateImageSource.h>
#include <ivq/ITK/EigenUtils.h>
#include <ivqML/Common/PCA.h>


namespace ivq
{
  namespace ITK
  {
    /**
     */
    template< class TOutputImage >
    class GenerateImageSource
      : public itk::GenerateImageSource< TOutputImage >
    {
    public:
      ITK_DISALLOW_COPY_AND_MOVE( GenerateImageSource );

      using Self         = GenerateImageSource;
      using Superclass   = itk::GenerateImageSource< TOutputImage >;
      using Pointer      = itk::SmartPointer< Self >;
      using ConstPointer = itk::SmartPointer< const Self >;

      using OutputImageType = typename Superclass::OutputImageType;
      using OutputImagePointer = typename Superclass::OutputImagePointer;
      using PixelType = typename Superclass::PixelType;
      using RegionType = typename Superclass::RegionType;
      using SpacingType = typename Superclass::SpacingType;
      using PointType = typename Superclass::PointType;
      using DirectionType = typename Superclass::DirectionType;
      using IndexType = typename Superclass::IndexType;
      using ReferenceImageBaseType = typename Superclass::ReferenceImageBaseType;
      using SizeType = typename Superclass::SizeType;
      using SizeValueType = typename Superclass::SizeValueType;

      static constexpr unsigned int NDimensions = Superclass::NDimensions;

    public:
      itkNewMacro( Self );
      itkTypeMacro( ivq::ITK::GenerateImageSource, itk::GenerateImageSource );


      itkGetConstMacro( NumberOfComponentsPerPixel, unsigned long long );
      itkSetMacro( NumberOfComponentsPerPixel, unsigned long long );

    protected:
      GenerateImageSource( )
        : Superclass( )
        {
        }
      virtual ~GenerateImageSource( ) override = default;

      virtual void GenerateOutputInformation( ) override
        {
          this->Superclass::GenerateOutputInformation( );
          std::cout << "----------> " << this->GetNumberOfOutputs( ) << std::endl;
          for( unsigned int n = 0; n < this->GetNumberOfOutputs( ); ++n )
          {
            OutputImageType* out = this->GetOutput( n );
            if( out != nullptr )
              out->SetNumberOfComponentsPerPixel( this->m_NumberOfComponentsPerPixel );
          } // end for
        }

      virtual void GenerateData( ) override
        {
          OutputImageType* out = this->GetOutput( );
          out->SetBufferedRegion( out->GetRequestedRegion( ) );
          out->Allocate( );

          std::cout << "------------------------------" << std::endl;
          out->Print( std::cout );
          std::cout << "------------------------------" << std::endl;
        }

    protected:
      unsigned long long m_NumberOfComponentsPerPixel { 1 };
    };
  } // end namespace
} // end namespace


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

      /* TODO
         virtual void GenerateOutputInformation( ) override
         {
         std::cout << "GenerateOutputInformation" << std::endl;

         TInImage* in = const_cast< TInImage* >( this->GetInput( ) );

         in->Update( );
         in->Print( std::cout );

         std::exit( 1 );


         TOutImage* out = this->GetOutput( );

         out->SetLargestPossibleRegion( in->GetLargestPossibleRegion( ) );
         out->SetRequestedRegion( in->GetRequestedRegion( ) );
         out->SetSpacing( in->GetSpacing( ) );
         out->SetOrigin( in->GetOrigin( ) );
         out->SetDirection( in->GetDirection( ) );
         }
      */

      virtual void CallCopyOutputRegionToInputRegion(
        typename TInImage::RegionType& srcRegion,
        const typename TOutImage::RegionType& destRegion
        ) override
        {
          this->Superclass::CallCopyOutputRegionToInputRegion( srcRegion, destRegion );
          std::cout << "*** uno ***" << std::endl;
        }
      
      virtual void CallCopyInputRegionToOutputRegion(
        typename TOutImage::RegionType& destRegion,
        const typename TInImage::RegionType& srcRegion
        ) override
        {
          this->Superclass::CallCopyInputRegionToOutputRegion( destRegion, srcRegion );
          std::cout << "*** dos ***" << std::endl;
        }

      virtual void GenerateData( ) override
        {
          std::cout << "GenerateData" << std::endl;

          const TInImage* in = this->GetInput( );

          auto X = ivq::ITK::ImageToMatrix( in ).transpose( );
          auto R = ivqML::Common::PCA< decltype( X ), TReal >( X, 0.95 );

          auto out = ivq::ITK::GenerateImageSource< TOutImage >::New( );
          out->SetReferenceImage( in );
          out->SetNumberOfComponentsPerPixel( R.second.cols( ) );
          out->UseReferenceImageOn( );

          std::cout << "a" << std::endl;
          out->GraftOutput( this->GetOutput( ) );
          std::cout << "b" << std::endl;
          out->Update( );
          std::cout << "c" << std::endl;
          this->GraftOutput( out->GetOutput( ) );
          std::cout << "d" << std::endl;

          std::cout << "----> " << this->GetOutput( )->GetNumberOfComponentsPerPixel( ) << std::endl;

          /* TODO
             TOutImage* out = this->GetOutput( );
             out->SetNumberOfComponentsPerPixel( R.second.cols( ) );
             out->SetBufferedRegion( out->GetRequestedRegion( ) );
             out->Allocate( );
             std::cout << out->GetNumberOfComponentsPerPixel( ) << std::endl;

             ivq::ITK::ImageToMatrix( out ) = R.second.transpose( );
            */
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
