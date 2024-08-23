// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__PCAImageFilter__h__
#define __ivqML__ITK__PCAImageFilter__h__

#include <itkImageToImageFilter.h>
#include <itkVectorImage.h>

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

      using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;

    public:
      itkNewMacro( Self );
      itkTypeMacro(
        ivqML::ITK::PCAImageFilter, itk::ImageToImageFilter
        );

      itkGetConstMacro( KeptInformation, TReal );
      itkSetMacro( KeptInformation, TReal );

      itkGetConstMacro( Mean, TMatrix );
      itkGetConstMacro( Rotation, TMatrix );
      itkGetConstMacro( Values, TMatrix );

    public:
      void SetNumberOfKeptDimensions( const unsigned int& i );

    protected:
      PCAImageFilter( );
      virtual ~PCAImageFilter( ) override = default;

      virtual void GenerateOutputInformation( ) override;
      virtual void GenerateData( ) override;

    private:
      PCAImageFilter( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;

    protected:

      TReal m_KeptInformation { 1 };

      TMatrix m_Mean;
      TMatrix m_Rotation;
      TMatrix m_Values;
    };
  } // end namespace
} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#  include <ivqML/ITK/PCAImageFilter.hxx>
#endif // ITK_MANUAL_INSTANTIATION

#endif // __ivqML__ITK__PCAImageFilter__h__

// eof - $RCSfile$
