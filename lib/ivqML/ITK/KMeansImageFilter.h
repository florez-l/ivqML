// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__KMeansImageFilter__h__
#define __ivqML__ITK__KMeansImageFilter__h__

#include <itkImage.h>
#include <itkImageToImageFilter.h>

namespace ivqML
{
  namespace ITK
  {
    /**
     */
    template< class _TInImage, class _TLabel = unsigned char, class _TReal = float >
    class KMeansImageFilter
      : public itk::ImageToImageFilter< _TInImage, itk::Image< _TLabel, _TInImage::ImageDimension > >
    {
    public:
      using TInImage  = _TInImage;
      using TLabel    = _TLabel;
      using TReal     = _TReal;
      using TOutImage = itk::Image< TLabel, TInImage::ImageDimension >;

      using Self         = KMeansImageFilter;
      using Superclass   = itk::ImageToImageFilter< TInImage, TOutImage >;
      using Pointer      = itk::SmartPointer< Self >;
      using ConstPointer = itk::SmartPointer< const Self >;

      using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;

    public:
      itkNewMacro( Self );
      itkTypeMacro(
        ivqML::ITK::KMeansImageFilter, itk::ImageToImageFilter
        );

      itkGetConstMacro( NumberOfMeans, unsigned long long );
      itkSetMacro( NumberOfMeans, unsigned long long );

      itkGetConstMacro( Means, TMatrix );

    protected:
      KMeansImageFilter( );
      virtual ~KMeansImageFilter( ) override = default;

      virtual void GenerateData( ) override;

    private:
      KMeansImageFilter( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;

    protected:
      unsigned long long m_NumberOfMeans { 2 };
      TMatrix m_Means;
    };
  } // end namespace
} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#  include <ivqML/ITK/KMeansImageFilter.hxx>
#endif // ITK_MANUAL_INSTANTIATION

#endif // __ivqML__ITK__KMeansImageFilter__h__

// eof - $RCSfile$
